""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import math
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from .utils import combine_tokens, token2feature, feature2token


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    # mae ViT-B/16-224 pre-trained model
    'vit_base_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch16_224_default': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-L/16-224 pre-trained model
    'vit_large_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-H/14-224 pre-trained model
    'vit_huge_patch14_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Interface_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None):
        super(Interface_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = x0 + x1
        return self.conv1x1(x0)

class VisionTransformerMM(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, search_size=384, template_size=192,
                 patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 search_number=1, template_number=1, use_checkpoint=False,
                 interface_type=None, interface_dim=8, instruct=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_list = [embed_dim]
        self.num_search = search_number
        self.num_template = template_number

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.patch_embed_interface = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search + self.num_patches_template, embed_dim))
        self.pos_embed_search = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.pos_embed_template = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # for multi-modal
        self.interface_type = interface_type
        '''interface parameters'''
        if self.interface_type in ['low-rank_add']:
            interface_blocks = []
            block_nums = depth
            for i in range(block_nums):
                if self.interface_type == 'low-rank_add':
                    interface_blocks.append(Interface_block(inplanes=embed_dim, hide_channel=interface_dim))
                else:
                    raise NotImplementedError
            self.interface_blocks = nn.Sequential(*interface_blocks)

            interface_norms = []
            for i in range(block_nums):
                interface_norms.append(norm_layer(embed_dim))
            self.interface_norms = nn.Sequential(*interface_norms)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.instruct = instruct
        if instruct:
            num_embeddings = 5
            self.prompt_embeddings = nn.Embedding(num_embeddings, embed_dim) # should be consistent with new tokens in decoder.instruct_tokens

        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed_search, std=.02)
        trunc_normal_(self.pos_embed_template, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, template_list, search_list, text_src, seq):
        num_template = len(template_list)
        num_search = len(search_list)
        if self.instruct:

            instruct_embedding = self.prompt_embeddings(seq).unsqueeze(1)

        z = torch.stack(template_list, dim=1)#(b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])#(bn,c,h,w)
        x = torch.stack(search_list, dim=1)#(b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])#(bn,c,h,w)

        # rgb image
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # multi-modal image
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]

        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)

        if self.interface_type in ['low-rank_add']:
            z_dte = self.patch_embed_interface(z_dte)
            x_dte = self.patch_embed_interface(x_dte)
            z_dte, x_dte = self.language_interface(z_dte, x_dte, text_src) # add language information
            z_rgb_feat = token2feature(self.interface_norms[0](z_rgb))
            x_rgb_feat = token2feature(self.interface_norms[0](x_rgb))
            z_dte_feat = token2feature(self.interface_norms[0](z_dte))
            x_dte_feat = token2feature(self.interface_norms[0](x_dte))
            z_feat = torch.cat([z_rgb_feat, z_dte_feat], dim=1)
            x_feat = torch.cat([x_rgb_feat, x_dte_feat], dim=1)
            z_feat = self.interface_blocks[0](z_feat)
            x_feat = self.interface_blocks[0](x_feat)
            z_dte = feature2token(z_feat)
            x_dte = feature2token(x_feat)
            x = x_rgb + x_dte
            z = z_rgb + z_dte
            z = z + self.pos_embed_template
            x = x + self.pos_embed_search
            x_dte = x_dte.reshape(-1, num_search * x_dte.size(1), x_dte.size(-1))
            z_dte = z_dte.reshape(-1, num_template * z_dte.size(1), z_dte.size(-1))
            z = z.reshape(-1, num_template * z.size(1), z.size(-1))
            x = x.reshape(-1, num_search * x.size(1), x.size(-1))
            len_x = x.size(1)
            len_z = z.size(1)
            xz = torch.cat([x, z], dim=1)
        else:
            raise ValueError('illegal interface_type')

        if self.instruct:
            xz = torch.cat([instruct_embedding, xz], dim=1)

        xz = self.pos_drop(xz)

        for i, blk in enumerate(self.blocks):   #batch is the first dimension.
            if i >= 1:
                if self.interface_type in ['low-rank_add']:
                    if self.instruct:
                        instruct_embedding = xz[:, 0, :].unsqueeze(1)
                        xz = xz[:, 1:, :]
                    xz_ori = xz
                    x = xz[:, :len_x, :]
                    z = xz[:, len_x:, :]
                    x = x.reshape(x.size(0)*num_search,-1,x.size(-1))
                    z = z.reshape(z.size(0)*num_template,-1,z.size(-1))
                    x_dte = x_dte.reshape(x_dte.size(0)*num_search,-1,x.size(-1))
                    z_dte = z_dte.reshape(z_dte.size(0)*num_template,-1,z.size(-1))
                    x_rgb_feat = token2feature(self.interface_norms[i](x))
                    z_rgb_feat = token2feature(self.interface_norms[i](z))
                    x_dte_feat = token2feature(self.interface_norms[i](x_dte))
                    z_dte_feat = token2feature(self.interface_norms[i](z_dte))
                    z_feat = torch.cat([z_rgb_feat, z_dte_feat], dim=1)
                    x_feat = torch.cat([x_rgb_feat, x_dte_feat], dim=1)
                    z_feat = self.interface_blocks[i](z_feat)
                    x_feat = self.interface_blocks[i](x_feat)
                    z_dte = feature2token(z_feat)
                    x_dte = feature2token(x_feat)
                    x_dte = x_dte.reshape(-1,num_search*x_dte.size(1),x_dte.size(-1))
                    z_dte = z_dte.reshape(-1,num_template*z_dte.size(1),z_dte.size(-1))
                    xz_dte = torch.cat([x_dte, z_dte], dim=1)
                    xz = xz_ori + xz_dte
                    if self.instruct:
                        xz = torch.cat([instruct_embedding, xz], dim=1)

            if self.use_checkpoint:
                xz = checkpoint.checkpoint(blk, xz)
            else:
                xz = blk(xz)

        xz = self.norm(xz) # B,N,C
        return xz

    def forward_features_rgb(self, template_list, search_list):
        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1)#(b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])#(bn,c,h,w)
        x = torch.stack(search_list, dim=1)#(b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])#(bn,c,h,w)

        x = self.patch_embed(x)
        z = self.patch_embed(z)


        z = z + self.pos_embed_template
        x = x + self.pos_embed_search

        # for multiple search region and template, go back
        z = z.reshape(-1,num_template * z.size(1),z.size(-1))
        x = x.reshape(-1,num_search * x.size(1),x.size(-1))

        len_x = x.size(1)
        len_z = z.size(1)

        xz = torch.cat([x, z], dim=1)

        xz = self.pos_drop(xz)

        for i, blk in enumerate(self.blocks):   #batch is the first dimension.
            if self.use_checkpoint:
                xz = checkpoint.checkpoint(blk, xz)
            else:
                print(i)
                xz = blk(xz)

        xz = self.norm(xz) # B,N,C
        return xz

    def forward(self, template_list, search_list, text_src, seq):
        xz = self.forward_features(template_list, search_list, text_src, seq)
        out=[xz]
        return out

    def forward_rgb(self, template_list, search_list):
        xz = self.forward_features_rgb(template_list, search_list)
        out=[xz]
        return out

    def language_interface(self, z_dte, x_dte, text_src):
        text_src = text_src.unsqueeze(1)
        x_dte = x_dte * text_src
        text_src_z = text_src.expand(-1,self.num_template,-1).reshape(text_src.size(0)*self.num_template,1,-1)
        z_dte = z_dte * text_src_z
        return z_dte, x_dte

@register_model
def vitmm_base_patch16(pretrained=False, pretrain_type='default',
                       search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformerMM(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_base_patch16_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type,
                        num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def vitmm_large_patch16(pretrained=False, pretrain_type='default',
                        search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformerMM(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_large_patch16_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def vitmm_huge_patch14(pretrained=False, pretrain_type='default',
                       search_size=364, template_size=182, **kwargs):
    patch_size = 14
    model = VisionTransformerMM(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_huge_patch14_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model,
                        pretrain_type, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

def load_pretrained(model, pretrain_type='default', cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=False):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        print("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    if pretrain_type == 'mae':
        state_dict = state_dict['model']

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        print('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            print('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            print('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if pretrain_type == "mae":
        pass
    elif num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']

    # adjust position encoding
    pe = state_dict['pos_embed'][:,1:,:]
    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0,3,1,2])  #b,c,h,w
    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search], align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0,2,3,1]),1,2)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template], align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_t = pe
    state_dict['pos_embed_template'] = pe_t
    state_dict['pos_embed_search'] = pe_s
    del state_dict['cls_token']
    del state_dict['pos_embed']

    model.load_state_dict(state_dict, strict=strict)

