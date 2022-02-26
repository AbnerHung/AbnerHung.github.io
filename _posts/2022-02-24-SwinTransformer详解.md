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

<div id="top"/>

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
            # BasicLayer作为程序中的stage，顺序与论文稍有不同
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

在每个stage开始前做降采样，用于缩小分辨率，调整通道数，形成层次化设计，一定程度节省算力。

而CNN则使用stride=2的卷积或池化来降采样。

```python
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # 0::2 偶数
        # 1::2 奇数
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) #  4维到2维的线性层，达到原patch通道c->2c，分辨率减半通道数加倍的效果。

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
```

![img](https://s2.loli.net/2022/02/26/g8koARqVYHIWNcZ.jpg)

### Basic Layer

程序实现的stage

```python
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth # swin transformer blocks 的个数
        self.use_checkpoint = use_checkpoint

        # 从0开始偶数位置block计算的是W-MSA；奇数位置block计算的是SW-MSA，且shift_size = window_size // 2
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
```

### Window Partition/Reverse

`window partition`函数是用于对张量划分窗口，指定窗口大小。将原本的张量从 `N H W C`, 划分成 `num_windows*B, window_size, window_size, C`，其中 `num_windows = H*W / window_size`，即窗口的个数。而`window reverse`函数则是对应的逆过程。这两个函数会在后面的`Block`和`Window Attention`用到。

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```



### Swin Transformer Block

这部分是程序的核心，涉及到相对位置编码，mask，window self-attention ，shifted window self-attention

先来看看Swin Transformer Block的整体，再仔细看看其他模块

Swin Transformer Block结构图：

![image-20220226155505740](https://s2.loli.net/2022/02/26/Gi7eFUbBHrACsqa.png)

Swin Transformer使用window self-attention 降低计算复杂度，为保证不重叠窗口直接有联系，使用shifted window self-attention重新计算了一遍窗口偏移后的自注意力，所以Swin Transformer Block都是成对出现的，即W-MSA和SW-MASA为一对，不同大小的block个数也都是偶数，不可能为奇数。

整个Block的流程是：

- 对特征图进行LayerNorm；

- 根据`self.shift_size`决定是否需要对特征图进行shift；

- 将特征图切成一个个window；

- 计算Attention， 通过`self.attn_mask`来区分是Window Attention还是 Shifted Window Attention；

- 将各个窗口合起来；
- 如果有shift操作，reverse shift恢复；
- dropout和残差连接
- 再经过一层LN+线性层，dropout和残差连接

```python
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 这是mask attention的相关代码
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # LN
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:   # 根据self.shift_size决定是否需要对特征图进行shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows 将特征图切成一个个window
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(se
                                        lf.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
```

### Window Attention

前面也提到并举例说明了，Swin Transformer相较于传统Transformer的全局计算注意力，它将注意力的计算限制在每个窗口内，大幅减少计算量。

原始公式[<sup>6</sup>](#origin_formula)：

$Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}})V$

Swin Transformer公式：

$Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}}+B)V$

可以看到公式上的区别只是加了一个相对位置编码

```python
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads #  每个注意力头对应的通道数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # 设置一个这样形状的可学习参数用于后续的位置编码

        # 相对位置编码 👇
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
		# 相对位置编码 👆
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 对q乘以一个缩放系数然后与k相乘，得到attn张量
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 根据相对位置编码索引选取，得到一个编码向量加到attn上
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 对应Swin Transformer Block中的SW-MSA
        # 不考虑mask的情况就是和transformer一样的softmax，dropout，与V矩阵相乘再经过一层全连接层和dropout
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
```

相对位置编码：

首先QK计算出来的的Attention张量形状为`(numWindows*B, num_heads, window_size*window_size,window_size*window_size)`

而对于Attention张量来说，**以不同元素为原点其他元素坐标是不同的**，如图，window_size=2为例

![image-20220226185803729](https://s2.loli.net/2022/02/26/UmDL3lY9hov2g8i.png)

首先用torch.arange和torch.meshgrid生成对应坐标, 以window_size=2为例

```python
coords_h = torch.arange(self.window_size[0])
coords_w = torch.arange(self.window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
"""
   (tensor([[0, 0],
           [1, 1]]), 
   tensor([[0, 1],
           [0, 1]]))
"""
```

堆叠起来展开为一个二维向量

```python
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
"""
tensor([[0, 0, 1, 1],
        [0, 1, 0, 1]])
"""
```

使用广播机制，分别在第一维，第二维，插入一个维度，进行广播相减

```python
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
#                     (2, Wh*Ww, 1)              (2, 1, Wh*Ww)
# (2, Wh*Ww, Wh*Ww)
```

得到 (2, Wh\*Ww, Wh\*Ww)的张量

加上偏移量，让索引从0开始

```python
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += self.window_size[1] - 1
```

对于(1, 2)和(2, 1)这组坐标，在二维上是不同的，但经过x, y坐标相加转换为一维偏移时是不可区分的，故使用一个乘法操作进行区分

```python
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
```

再在最后一维上进行求和，展开成一个一维坐标，并注册为一个不参与网络学习的变量

```python
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
self.register_buffer("relative_position_index", relative_position_index)
```

![image-20220226201727119](https://s2.loli.net/2022/02/26/vrkTJU9Lc8jxFX6.png)

后面计算QK值时，每个QK值都要加上对应的相对位置编码

这个relative_position_index的作用就是根据最终索引值找到对应可学习的相对位置编码，以M=2的窗口为例，这个值的数值范围是[0,8], 所以相对位置编码可以由3\*3的矩阵表示，推广至窗口大小为M即$(2M-1)*(2M-1)$的矩阵。

![image-20220226201654464](https://s2.loli.net/2022/02/26/3xSjymbgcnNBq7V.png)

继续以window_size=2为例，即M=2，计算位置1对应的$M^2$个QK值时，应用的relative_position_index = [ 4, 5, 7, 8]，（$M^2$个，这里是4个）对应的数据就是上图索引4，5，7，8位置对应的$M^2$维数据，$relative\_position.shape=(M^2*M^2)$

### Shifted Window Attention

前面的Window Attention是在每个窗口下计算注意力的，为了更好地与其他window建立联系，Swin Transformer引入了shifted window操作

![image-20220226205655202](https://s2.loli.net/2022/02/26/ZJi1uCdQUVcxNrI.png)

左边是没有重叠窗口的Window Attention，右边是将窗口进行移位的Shift Window Attention，移位后的窗口可以包含原本相邻窗口的元素，但也引入了一个问题，即window的个数由原来的4个变为了9个（以上图为例）

在实际代码中，他们是通过对特征图移位，并给Attention设置mask来实现。能在保持原有window个数下，最后计算结果等价。

![image-20220226210120061](https://s2.loli.net/2022/02/26/XTCsKAGeR729Yrv.png)

### 特征图移位

代码里使用torch.roll来实现

`roll(x, shifts=-1, dims=1)`    `roll(x, shifts=-1, dims=2)`

![image-20220226210519046](https://s2.loli.net/2022/02/26/QGNALEq7gbSnFkz.png)

如果需要`reverse cyclic shift`的话只需把参数`shifts`设置为对应的正数值。

### Attention Mask

通过设置合理的mask可以让Shifted Window Attention在与Window Attention相同窗口的个数下达到等价的计算效果。

如上图，在(5,3)\(7,1)\(8,6,2,0)组成的新窗口中，我们希望在计算Attention的时候，让具相同index的QK进行计算，而忽略不同index的QK计算结果。根据自注意力的计算公式，最后需要进行Softmax操作，如果我们对不同编码位置见计算的self-attention结果加上-100，在Softmax计算过程中就可以达到归零的效果。

mask的形式如下图：

![image-20220226212443486](https://s2.loli.net/2022/02/26/vDizV8fu4e5r6YF.png)

mask与相对位置编码有着一样的形式，$mask.shape=(num\_windows,M^2,M^2)$

在上面介绍的Block的代码中有这部分的代码实现

```python
# 这是mask attention的相关代码
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
      # 👆原代码就是使用-100去mask掉不用计算的部分
```

前面Window Attention代码里前向传播也有这么一段：

```python
 		if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
```

  以上全部就是Swin Transformer模型的全部代码了, 现在再[回到前面](#top)再看一看。

## V2

### 前言

论文中提到Swin Transformer V2[<sup>7</sup>](#swin-transformer-2)和V1一样，最终目的都是为了能够联合建模NLP和CV模型。

V2的直接目标是得到一个大规模的预训练模型，可以应用到其他视觉任务并取得高精度。

NLP目前的模型参数已经达到了千亿级别并且出现了像BERT这种成功的预训练模型可以适配不同的NLP任务；CV目前最主要的一个问题就是模型规模不够大，ViT-G参数量也只有不到20亿，并且所有大规模的视觉模型都只应用于图片分类任务。

为了统一CV和NLP，CV模型存在两个问题：

- 模型规模不够大
- 预训练模型与下游任务图片分辨率，窗口大小不适配

### 解决方法

对于模型不够大，提出了**post-norm and cosine similarity**；

对于模型不适配，提出了**Continuous relative position bias 和 Log-spaced coordinates**

### 对于模型不够大

论文中提出了SwinV2-G: C=512, layer numbers={2,2,42,2}，通过增加处理数据的维度(192-->512)和Transformer层数(18-->42)增大模型规模，但是会产生一个问题：模型训练不稳定甚至不能完成训练。通过对模型每一层输出的分析，发现随着模型变深，没层的输出是不断变大的，随着层数增加，输出值不断累加，深层输出和浅层输出幅值差很多，导致训练过程不稳定(Figure 2)，解决这个问题就是要稳定每一层的输出，其幅值要稳定才会使得训练过程稳定。

![image-20220226221944296](https://s2.loli.net/2022/02/26/nE5mZMVUK3yw91v.png)

为了稳定深层和浅层的输出，Swin V2 提出了post-norm 和 cosine similarity

### post-norm

![image-20220226222040025](https://s2.loli.net/2022/02/26/MYn5XUj9LbtI6wZ.png)

post-norm 就是把之前通用ViT中的Transformer block中的Layer Norm层从Attention层前面挪到后面，这么做的好处就是计算Attention之后会对输出进行归一化操作，稳定输出值

### cosine similarity

ViT中Transformer block计算Attention是采用dot(Q,K)的操作，在Swin V2中将其替换为了cosine(Q,K)/τ，τ是可学习参数，block之间不共享。cosine自带normalization操作，会进一步稳定Attention输出值

通过post-norm和cosine similarity操作将block的输出稳定在可接受范围内, 帮助模型进行稳定的训练。

### 对于模型不适配

模型不适配主要是图片分辨率和窗口大小的问题，在预训练过程中为了节约训练成本，采用的训练图片分辨率通常比较低(192/224)，但是很多下游任务要求的图片分辨率远大于训练分辨率的(1024/1536)，这就导致一个问题，将预训练模型迁移到下游任务时为了适配输入图片分辨率势必要改变窗口大小，改变窗口大小就会改变patch数量，改变patch数量就会导致相对位置编码偏置发生变化。e.g: 8×8 window size 变为 16×16 window size，其相对编码坐标就从[-7,7] --> [-15,15]，多出来的位置如何计算相对位置编码？ 之前通用的方法是采用二项三次差值，但是效果次优，不够灵活。

为了解决这个问题，Swin V2 提出了 Continuous relative position bias 和 Log-spaced coordinates

### Continuous relative position bias

V1使用二项三次差值的方法生成偏置B；而如Figure1 V2示意图的左下角红色的部分，Swin V2直接用两层MLP（Meta network）来适应生成相对位置偏置。

### Log-spaced coordinates

patch数量变化之后需要将相对位置编码外推，在例子中的外推率是1.14×，我们希望的是降低这个比率，外推的越小越好，毕竟Meta network没见过外推特别多的输入，为了保证相对位置编码的准确性，需要将外推控制在一个可接受范围内。

Log-spaced coordinates出现，将线性变换转换为了对数变换：

![image-20220226223004475](https://s2.loli.net/2022/02/26/pYNwrtQIhgfvEqM.png)

![image-20220226223101072](https://s2.loli.net/2022/02/26/vE6I2gyMPHRun1f.png)

![image-20220226223111616](https://s2.loli.net/2022/02/26/RDtxo5Km49FV2Uw.png)





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
<div id="origin_formula"/>
- [6] [超详细图解Self-Attention](https://zhuanlan.zhihu.com/p/410776234)

  <div id="swin-transformer-2"/>

- [7] [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

  

- [8] [Swin Transformer V2 论文解析](https://zhuanlan.zhihu.com/p/445876985)

  

- [9] [图解Swin Transformer](https://zhuanlan.zhihu.com/p/367111046)
