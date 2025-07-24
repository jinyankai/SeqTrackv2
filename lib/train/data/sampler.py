import random
import torch.utils.data
from torch.utils.data import Sampler

from lib.train.dataset.our_data import MyDataset
from lib.utils import TensorDict
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import os

def no_processing(data):
    return data

def get_sampling_mode(epoch):
    if epoch < 10:  # 阶段一
        # 50% Causal, 50% Order
        return random.choices(['causual', 'order'], weights=[0.5, 0.5], k=1)[0]
    elif 10 <= epoch < 100:  # 阶段二
        # 20% Causal, 20% Order, 60% Trident
        return random.choices(['causual', 'order', 'trident'], weights=[0.2, 0.2, 0.6], k=1)[0]
    else:  # 阶段三
        # 10% Causal, 10% Order, 40% Trident, 40% STARK
        return random.choices(['causual', 'order', 'trident', 'stark'], weights=[0.1, 0.1, 0.4, 0.4], k=1)[0]



class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5, max_query_len=30,
                 bert_model='bert-base-uncased', bert_path=None, multi_modal_language=False):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - 'causal', 'interval', or 'order'.
            train_cls - this is for Stark-ST, should be False for SeqTrack.

        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.multi_modal_language = multi_modal_language
        if multi_modal_language:
            self.max_query_len = max_query_len
            if bert_path is not None and os.path.exists(bert_path):
                self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)
    def update_sampler_mode(self, epoch):
        """ Update the frame sampling mode based on the current epoch """
        self.frame_sample_mode = get_sampling_mode(epoch)
        print(f"Updated frame sample mode to: {self.frame_sample_mode}")

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        count_valid = 0
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)


            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causual':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:

                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)

                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])

                        if prev_frame_ids is None:
                            gap_increase += 5
                            if gap_increase > 1000:
                                print("too large image gap, check the sampler, current gap: "+str(gap_increase))
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5
                        if gap_increase > 1000:
                            print("too large image gap, check the sampler, current gap: " + str(gap_increase))

                elif self.frame_sample_mode == "order":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_order(visible)
                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames
                
                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})


                # tokenize language
                if self.multi_modal_language:
                    # nlp = template_anno['nlp'][0]
                    nlp = template_anno.get("nlp", None)
                    if nlp is not None:
                        nlp = nlp[0]
                        nlp_token_ids, nlp_token_masks = self.extract_token_from_nlp(nlp, self.max_query_len)
                        data['nl_token_ids'] = nlp_token_ids
                        data['nl_token_masks'] = nlp_token_masks
                        


                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except Exception as e:
                print(f"data sampler bug: {e}")
                valid = False

            count_valid += 1
            if count_valid > 200:
                print("too large count_valid, check the sampler, current count_valid: "+str(count_valid))

        return data

    def show(self, data, strr, i, modality):
        image = data[strr+'_images'][i]
        if modality == 'rgb':
            image = image[:3,:,:]
        else:
            image = image[3:, :, :]
        _, H, W = image.shape
        import cv2
        x1, y1, w, h = data[strr+'_anno'][i]
        x1, y1, w, h = int(x1*W), int(y1*H), int(w*W), int(h*H)
        image_show = image.permute(1,2,0).numpy()
        max = image_show.max()
        min = image_show.min()
        image_show = (image_show-min) * 255 / (max-min)
        image_show = np.ascontiguousarray(image_show.astype('uint8'))
        cv2.rectangle(image_show, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        cv2.imshow(strr+str(i)+modality, image_show)
        if cv2.waitKey() & 0xFF == ord('q'):
            pass

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        #add by chenxin to debug
        count = 0
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
            count += 1
            if count > 200:
                print("too large count, check the sampler, current count: " + str(count))
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_order(self, visible):
        # get template and search ids in an 'order' manner, the template and search regions are arranged in chronological order
        frame_ids = []
        gap_increase = 0
        while (None in frame_ids) or (len(frame_ids)==0):
            base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=0,
                                                     max_id=len(visible))
            frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames+self.num_search_frames,
                                                      min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                      max_id=base_frame_id[0] + self.max_gap + gap_increase)
            if (frame_ids is None) or (None in frame_ids):
                gap_increase += 5
                if gap_increase > 1000:
                    print("too large image gap, check the sampler, current gap: " + str(gap_increase))
                continue
            if torch.rand(1) < 0.5:
                frame_ids.sort(reverse=True)
                template_frame_ids = frame_ids[0:self.num_template_frames]
                search_frame_ids = frame_ids[self.num_template_frames:]
            else:
                frame_ids.sort(reverse=False)
                template_frame_ids = frame_ids[0:self.num_template_frames]
                search_frame_ids = frame_ids[self.num_template_frames:]
            # Increase gap until a frame is found
            gap_increase += 5
            if gap_increase > 1000:
                print("too large image gap, check the sampler, current gap: " + str(gap_increase))
        return template_frame_ids, search_frame_ids

    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask


if __name__ == '__main__':
    import sys
    sys.path.append('/home/jinyankai/PycharmProject/SeqTrackv2/lib/train/dataset')
    from our_data import MyDataset
    NUM_SEQUENCES = 1
    NUM_FRAMES = 3000
    BATCH_SIZE = 4
    SAMPLES_PER_EPOCH = 5
    NUM_TEMPLATE_FRAMES = 1
    NUM_SEARCH_FRAMES = 1
    dataset_root = '/home/jinyankai/PycharmProject/SeqTrackv2/data'
    try:
        datasets = MyDataset(name='my_multi_sequence_dataset', root_path=dataset_root)
        sampler = TrackingSampler(datasets=[datasets] , p_datasets=None,
                samples_per_epoch=SAMPLES_PER_EPOCH,
                max_gap=10,
                num_search_frames=NUM_SEARCH_FRAMES,
                num_template_frames=NUM_TEMPLATE_FRAMES)
        print("成功实例化 TrackingSampler。")

        # 5. 实例化 DataLoader
        # PyTorch 的默认 collate_fn 足够智能，可以处理这种结构
        data_loader = torch.utils.data.DataLoader(
            sampler,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        print(f"成功实例化 DataLoader with batch_size={BATCH_SIZE}。")

        # 6. 迭代并检查数据
        print("\n--- 开始测试数据加载 ---")
        for i, data in enumerate(data_loader):
            if i >= 3:  # 只测试前3个批次
                break

            print(f"\n[批次 {i + 1}]")
            print("  数据包键:", list(data.keys()))

            # 检查模板数据
            # 期望形状: [Batch, NumFrames, Height, Width, Channels]
            # 由于我们把NumFrames=1的数据堆叠起来，所以形状是 [B, H, W, C]
            template_images = data['template_images']
            template_annos = data['template_anno']
            print(f"  - template_images shape: {template_images[0].shape} (Batch, H, W, Channels)")
            print(f"  - template_images dtype: {template_images[0].dtype}")
            print(f"  - template_anno shape:   {template_annos[0].shape} (Batch, NumFrames, 4)")
            print(f"  - template_anno dtype:   {template_annos[0].dtype}")

            # 检查搜索数据
            search_images = data['search_images']
            search_annos = data['search_anno']
            print(f"  - search_images shape:   {search_images[0].shape} (Batch, H, W, Channels)")
            print(f"  - search_images dtype:   {search_images[0].dtype}")
            print(f"  - search_anno shape:     {search_annos[0].shape} (Batch, NumFrames, 4)")
            print(f"  - search_anno dtype:     {search_annos[0].dtype}")
            print(f"  - dataset:               {data['dataset']}")

        print("\n--- 测试成功 ---")
        print("输出的数据形状符合预期，MyDataset 和 TrackingSampler 看起来已正确适配。")

    except Exception as e:
        print(f"\n--- 测试失败 ---")
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()

