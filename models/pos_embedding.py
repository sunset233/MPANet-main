# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Attention(nn.Module):
#     def __init__(self,
#                  dim,   # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, x):
#         # [batch_size, num_patches + 1, total_embed_dim]
#         B, N, C = x.shape
#
#         # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#         Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#         This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#         the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#         See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#         changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#         'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     # drop_prob是进行droppath的概率
#     keep_prob = 1 - drop_prob
#     # work with diff dim tensors, not just 2D ConvNets
#     # 在ViT中，shape是(B,1,1),B是batch size
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     # 按shape,产生0-1之间的随机向量,并加上keep_prob
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     # 向下取整，二值化，这样random_tensor里1出现的概率的期望就是keep_prob
#     random_tensor.floor_()  # binarize
#     # 将一定图层变为0
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class Mlp(nn.Module):
#     """
#     MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         # 将每个样本的每个通道的特征向量做归一化
#         # 也就是说每个特征向量是独立做归一化的
#         # 我们这里虽然是图片数据，但图片被切割成了patch，用的是语义的逻辑
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         # 全连接，激励，drop，全连接，drop,若out_features没填，那么输出维度不变。
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#     def forward(self, x):
#         # 最后一维归一化，multi-head attention, drop_path
#         # (B, N, C) -> (B, N, C)
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         # (B, N, C) -> (B, N, C)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size)
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(Block, self).__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F



class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out
class ViTModule(nn.Module):
    '''
        obtaining baseline feature from backbone
        covering resnet50 and hrnet-resnet50
        now use vision transformer to extract discriminative feature from network
        procedure:  features --> Norm --> MHSA --> concat --> Norm --> MLP --> concat

    '''
    def __init__(self, num_features):
        super(ViTModule, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        planes = 2048
        self.MHSA = MHSA(num_features, width=24, height=8, heads=4)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(x.shape)
        feat = self.MHSA(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = feat + x
        return feat
if __name__ == '__main__':
    x = torch.randn(32, 2048, 18, 9)
    model = ViTModule(2048)
    feat = model(x)
    print(feat.shape)