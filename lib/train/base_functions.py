import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, Imagenet1k,our_data
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent, LasHeR, DepthTrack
from lib.train.dataset import Otb99_lang, Tnl2k, RefCOCOSeq
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.train.dataset.our_data import MyDataset
from lib.utils.misc import is_main_process

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search':getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.multi_modal_vision = getattr(cfg.DATA, "MULTI_MODAL_VISION", False)
    settings.multi_modal_language = getattr(cfg.DATA, "MULTI_MODAL_LANGUAGE", False)
    if settings.multi_modal_language:
        settings.max_query_len = cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN
        settings.bert_model = cfg.MODEL.LANGUAGE.TYPE
        settings.bert_path = cfg.MODEL.LANGUAGE.VOCAB_PATH
    else:
        settings.max_query_len = None
        settings.bert_model = None
        settings.bert_path = None
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        settings.fix_norm = True
    else:
        settings.fix_norm = False


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full",
                        "COCO17", "VID", "TRACKINGNET", "IMAGENET1K",
                        "DepthTrack_train", "DepthTrack_val", "LasHeR_all", "LasHeR_train","LasHeR_val", "VisEvent",
                        "REFCOCOG", "TNL2K_train", "OTB99_train", "MY_DATA"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader,
                                           multi_modal_vision=settings.multi_modal_vision,
                                           multi_modal_language=settings.multi_modal_language))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language
                                            ))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader,
                                               multi_modal_vision=settings.multi_modal_vision,
                                               multi_modal_language=settings.multi_modal_language
                                               ))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader,
                                          multi_modal_vision=settings.multi_modal_vision,
                                          multi_modal_language=settings.multi_modal_language
                                          ))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader,
                                                 multi_modal_vision=settings.multi_modal_vision,
                                                 multi_modal_language=settings.multi_modal_language
                                                 ))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader,
                                            multi_modal_vision=settings.multi_modal_vision,
                                            multi_modal_language=settings.multi_modal_language
                                            ))
        if name == "IMAGENET1K":
            datasets.append(Imagenet1k(settings.env.imagenet1k_dir, image_loader=image_loader))
        if name == "DepthTrack_train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='train',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "DepthTrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, dtype='rgbcolormap', split='val',
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "LasHeR_all":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='all',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language
                                   ))
        if name == "LasHeR_train":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='train',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language
                                   ))
        if name == "LasHeR_val":
            datasets.append(LasHeR(settings.env.lasher_dir, dtype='rgbrgb', split='val',
                                   multi_modal_vision=settings.multi_modal_vision,
                                   multi_modal_language=settings.multi_modal_language
                                   ))
        if name == "VisEvent":
            datasets.append(VisEvent(settings.env.visevent_dir, dtype='rgbrgb', split='train',
                                     multi_modal_vision=settings.multi_modal_vision,
                                     multi_modal_language=settings.multi_modal_language
                                     ))
        if name == "REFCOCOG":
            datasets.append(RefCOCOSeq(settings.env.refcoco_dir, split="train", image_loader=image_loader,
                                       name="refcocog", splitBy="google",
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "TNL2K_train":
            datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader,
                                  multi_modal_vision=settings.multi_modal_vision,
                                  multi_modal_language=settings.multi_modal_language
                                  ))
        elif name == "OTB99_train":
            datasets.append(Otb99_lang(settings.env.otb99_dir, split='train', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language
                                       ))
        if name == "MY_DATA":
            datasets.append(MyDataset("MY_DATA", root_path=settings.env.our_data_dir,image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language))

    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          multi_modal_language=settings.multi_modal_language,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, max_query_len=settings.max_query_len,
                                            bert_model=settings.bert_model, bert_path=settings.bert_path,
                                            multi_modal_language=settings.multi_modal_language
                                            )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n or "interface" in n and p.requires_grad]},
        ]
        for n, p in net.named_parameters():
            if ("prompt" not in n) and ("interface" not in n):
                p.requires_grad = False

        # if is_main_process():
        #     print("Learnable parameters are shown below.")
        #     for n, p in net.named_parameters():
        #         if p.requires_grad:
        #             print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]
        # if is_main_process():
        #     print("Learnable parameters are shown below.")
        #     for n, p in net.named_parameters():
        #         if p.requires_grad:
        #             print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "cosine":
        # Cosine Annealing: Smoothly decays LR following a cosine curve.
        # Required cfg params:
        #   - cfg.TRAIN.SCHEDULER.T_MAX (int): Maximum number of iterations (usually total epochs).
        #   - cfg.TRAIN.SCHEDULER.ETA_MIN (float, optional): Minimum learning rate. Defaults to 0.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=cfg.TRAIN.SCHEDULER.T_MAX,
                                                      eta_min=getattr(cfg.TRAIN.SCHEDULER, 'ETA_MIN', 0))

    elif cfg.TRAIN.SCHEDULER.TYPE == "cosine_restarts":
        # Cosine Annealing with Warm Restarts: Restarts the cosine schedule every T_0 epochs.
        # Required cfg params:
        #   - cfg.TRAIN.SCHEDULER.T_0 (int): Number of epochs for the first restart.
        #   - cfg.TRAIN.SCHEDULER.T_MULT (int, optional): A factor to increase T_i after a restart. Defaults to 1.
        #   - cfg.TRAIN.SCHEDULER.ETA_MIN (float, optional): Minimum learning rate. Defaults to 0.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                T_0=cfg.TRAIN.SCHEDULER.T_0,
                                                                T_mult=getattr(cfg.TRAIN.SCHEDULER, 'T_MULT', 1),
                                                                eta_min=getattr(cfg.TRAIN.SCHEDULER, 'ETA_MIN', 0))
                                                                
    elif cfg.TRAIN.SCHEDULER.TYPE == 'exponential':
        # Exponential LR: Decays the LR by gamma every epoch.
        # Required cfg params:
        #   - cfg.TRAIN.SCHEDULER.GAMMA (float): The multiplicative factor.
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.TRAIN.SCHEDULER.GAMMA)

    elif cfg.TRAIN.SCHEDULER.TYPE == 'plateau':
        # ReduceLROnPlateau: Reduces LR when a metric has stopped improving.
        # **SPECIAL USAGE**: Call scheduler.step(metric) in your validation loop.
        # Required cfg params:
        #   - cfg.TRAIN.SCHEDULER.MODE ('min' or 'max'): 'min' for loss, 'max' for accuracy.
        #   - cfg.TRAIN.SCHEDULER.FACTOR (float): Factor by which the learning rate will be reduced.
        #   - cfg.TRAIN.SCHEDULER.PATIENCE (int): Number of epochs with no improvement to wait.
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      mode=cfg.TRAIN.SCHEDULER.MODE,
                                                      factor=cfg.TRAIN.SCHEDULER.FACTOR,
                                                      patience=cfg.TRAIN.SCHEDULER.PATIENCE,
                                                      verbose=True)

    elif cfg.TRAIN.SCHEDULER.TYPE == 'onecycle':
        # OneCycleLR: A powerful scheduler that varies LR over one cycle.
        # **SPECIAL USAGE**: Call scheduler.step() after every BATCH, not every epoch.
        # Required cfg params:
        #   - cfg.TRAIN.SCHEDULER.MAX_LR (float): Upper learning rate boundary.
        #   - cfg.TRAIN.SCHEDULER.STEPS_PER_EPOCH (int): Number of batches in an epoch.
        #   - cfg.TRAIN.SCHEDULER.EPOCHS (int): Total number of epochs.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                               max_lr=cfg.TRAIN.SCHEDULER.MAX_LR,
                                               steps_per_epoch=cfg.TRAIN.SCHEDULER.STEPS_PER_EPOCH,
                                               epochs=cfg.TRAIN.SCHEDULER.EPOCHS)
    
    elif cfg.TRAIN.SCHEDULER.TYPE == 'none':
        # No scheduler, constant learning rate.
        lr_scheduler = None

    else:
        raise ValueError(f"Unsupported scheduler: {cfg.TRAIN.SCHEDULER.TYPE}")
    return optimizer, lr_scheduler
