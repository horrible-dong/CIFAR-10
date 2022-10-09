import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_features, num_heads, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.attn = Attention(in_features, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.norm2 = nn.LayerNorm(in_features)
        self.mlp = Mlp(in_features=in_features, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(in_features)
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_features)

    def forward(self, x):
        x_ = x
        x = self.conv(x)
        x += x_
        x = self.relu(x)
        x = self.bn(x)

        return x


class MyNet(nn.Module):
    def __init__(self, c0=3, c1=64, c2=128, c3=256, c4=512, depth=4, heads=4, num_classes=10, drop_ratio=0.,
                 attn_drop_ratio=0.):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(c1),
            ResidualBlock(c1),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c2),
            ResidualBlock(c2),
            ResidualBlock(c2),
            nn.Conv2d(c2, c3, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c3),
            ResidualBlock(c3),
            ResidualBlock(c3),
            nn.Conv2d(c3, c4, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c4),
            ResidualBlock(c4),
            ResidualBlock(c4),
        )

        self.blocks = nn.Sequential(*[
            AttentionBlock(in_features=c4, num_heads=heads, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(depth)
        ])

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(c4 * 4, num_classes)

        self.apply(_init_weights)

    def forward(self, x):
        x = self.conv(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H * W, C]

        for block in self.blocks:
            x = block(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
