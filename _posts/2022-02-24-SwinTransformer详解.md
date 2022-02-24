---
redirect_from: /_posts/2022-02-24-SwinTransformer详解.md/
title: SwinTransformer详解
tags:
  - Python
  - AI
  - ComputerVision
  - transformer
---

# Swin Transformer[<sup>1</sup>](#swin-transformer-1)

## V1

### 前言

**更好地将语言模型建模到视觉**是近几年的一大热点，**ViT**[<sup>2</sup>](#vit)，**DETR**[<sup>3</sup>](#detr)等模型将Transformer引入视觉领域并取得了不错的效果。ViT直接把图片划分成patch，把patch看作word，将图片建模成sentence；DETR借助CNN提取特征，将Transformer当作neck。

而目前Transformer应用于图像领域的两大挑战：

- 视觉实体变化大，在不同场景下视觉Transformer性能未必很好
- 图像分辨率高，像素点多，基于自注意力机制的Transformer的计算复杂度是token数量的平方[<sup>4</sup>](#transformer-complexity)，若将每个像素点作为一个token，其计算量巨大

为解决上述问题，出现了包含**滑窗操作**，具有**层级设计**及的Swin Transformer

### 解决方法

1. 使用分层结构，使得模型能灵活处理不同尺度的图片。

   ![image-20220224191303395](https://s2.loli.net/2022/02/24/6tVnJGmuqQ9aFwO.png)

   

2. 使用window self-attention，降低计算复杂度

   简单地说，传统Transformer都是基于全局来计算注意力的，而Swin Transformer将注意力计算限制在每个window里，进而减少计算量。

   

   举个例子，设一张图片共有$h*w$个patches，每个窗口包括$n*n$个patches。

   那么传统Transformer的self-attention计算复杂度为${hw}^2$

   而Swin Transformer计算复杂度为窗口计算复杂度\*窗口数量

   窗口计算复杂度为${(n^2)}^2$, 窗口数量为$\frac {h*w}{n^2}$

   故Swin Transformer计算复杂度为$\frac{h*w}{n^2}*(n^2)^2=h*w*n^2$ (其中$n^2$是每个窗口包含的patches数，是确定值)

   

下面开始详细看看：

### 整体结构

![image-20220224193823861](https://s2.loli.net/2022/02/24/HJLlIS6fURaX39T.png)

一共4个stage，每个stage由Patch Merging 和 Swin Transformer Block组成，逐stage缩小特征图分辨率，像CNN一样扩大感受野。

源码实现逻辑如下图[<sup>5</sup>](#pic)

![image-20220224195008866](https://s2.loli.net/2022/02/24/LY1o4FXHmZTwg7x.png)

- 首先做了一个Patch Embedding，将图片切成一个个图块儿，并嵌入到embedding
- 与ViT不同的是，在输入时给embedding进行绝对位置编码在Swin Transformer中作为一个可选项，并且在计算Attention时做了一个相对位置编码
- 每个stage由多个block和Patch Merging组成
- Block具体结构如上上图中的b，主要由LN 、MLP、Window Attention和Shifted Window Attention组成
- ViT单独加入一个可学习参数作为分类的参数，而Swin Transformer直接平均池化，输出分类

```python
class SwinTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        #  在这里做了一个patch embedding，将图片切割成不重叠的patches，并且嵌入embedding
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 这里在输入时给embedding进行位置编码作为一个可选项
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Block和PatchMergigng
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],# 随机深度衰减
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,# PatchMerging
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 这里使用平均池化，输出分类
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
```

### Patch Embedding

对输入图片进入Block前的处理，将图片切成一个个$patch\_size * patch\_size*channels$大小的patches，然后嵌入向量（reshape成$n*(patch\_size^2*channels)\ where\ n=\frac{h*w}{patch\_size^2}$的patches块儿, 再通过线性变换将patches投影到维度为D的空间上, 也就是直接将原来大小为$h*w*channels$的二维图像展平成n个大小为$patch\_size^2*channels$的一维向量)

可以通过二维卷积层，将stride，kernel size设置为patch_size大小。

这里的处理和ViT基本是一样的。

卷积输出计算公式：$\frac{n+2p-f}{s}+1$

 将图像宽高带入分别得到

$\frac{h+0-p}{p}+1=\frac{h}{p}$, $\frac{w+0-p}{p}+1=\frac{w}{p}$

二者相乘即$\frac{h*w}{p^2}=n$, 这就等价于将输入图像划分成n个大小为$p^2*c$ 的patches

```python
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # -> (img_size, img_size)
        patch_size = to_2tuple(patch_size) # -> (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
```

### Patch Merging



# Reference
<div id="swin-transformer-1"></div>
- [1] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
<div id="vit"/>
- [2] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
<div id="detr"/>
- [3] [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
<div id="transformer-complexity"/>
- [4] [Transformer/CNN/RNN的对比（时间复杂度，序列操作数，最大路径长度）](https://zhuanlan.zhihu.com/p/264749298)
<div id="pic"/>
- [5] [Swin Transformer 论文详解及程序解读](https://zhuanlan.zhihu.com/p/401661320)
