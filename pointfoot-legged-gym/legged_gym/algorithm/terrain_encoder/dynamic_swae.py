import os
import random
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch import nn, Tensor
from torchvision import transforms
from torch.autograd import Variable
from torch import distributions as dist
from torch import nn
from typing import Callable, List, Any, Optional, Sequence, Type

from legged_gym.algorithm.terrain_encoder.swae_model import BaseVAE

class DynamicSWAE(BaseVAE):
    """
    SWAE 的动态多分辨率版本：
    - 输入 H=W∈{16,32,64}，自动决定下采样层数 K，使编码空间落到 2x2。
    - 64x64 -> K=5，32x32 -> K=4，16x16 -> K=3（每层 stride=2）。
    - 解码只用 K 次上采样回到原尺寸（实现为 (K-1) 个反卷积块 + 1 个 final_up）。
    其余接口与原 SWAE 尽量保持一致。
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List[int] = None,
                 reg_weight: int = 100,
                 wasserstein_deg: float = 2.,
                 num_projections: int = 200,
                 projection_dist: str = 'normal',
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]  # 每层都会 stride=2 下采样

        # --------- Encoder: 用 ModuleList 便于按需截断 ----------
        self.enc_blocks = nn.ModuleList()
        ch = in_channels
        for h_dim in hidden_dims:
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            ch = h_dim

        # 编码末端的全连接映射到 z：用 LazyLinear 以适配不同输入尺寸（C*H*W 变化）
        self.fc_z = nn.LazyLinear(latent_dim)

        # --------- Decoder: 反卷积块（固定最大深度，运行时按需截取） ----------
        self.hidden_dims = hidden_dims[:]  # 记录以便查通道
        dec_dims = list(reversed(hidden_dims))  # e.g. [512, 256, 128, 64, 32]

        self.dec_blocks = nn.ModuleList()
        for i in range(len(dec_dims) - 1):
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_dims[i], dec_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dec_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        # 最后一次上采样（第 K 次），通道保持不变，再用 conv 到 1 通道
        self.final_up = nn.ConvTranspose2d(dec_dims[-1], dec_dims[-1],
                                           kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(dec_dims[-1], out_channels=1, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()

        # 解码起始的线性层（latent -> Ck*4）按 Ck 不同而不同，做成缓存字典
        self.decoder_input_map = nn.ModuleDict()

        # 运行期形状/通道缓存
        self._in_hw: Optional[tuple[int, int]] = None
        self._enc_out_c: Optional[int] = None

    # -------------------- 工具方法 --------------------

    @staticmethod
    def _num_down_to_2x2(H: int, W: int) -> int:
        """
        计算需要下采样的层数 K，使 H,W 经 K 次 /2 变为 2。
        约束：H=W 且 H ∈ {16,32,64}
        """
        if H != W or H not in (16, 32, 64):
            raise ValueError(f"Only square inputs with size 16/32/64 supported, got {(H, W)}")
        # H = 2^(K+1)  =>  K = log2(H) - 1
        return int(np.log2(H) - 1)

    # -------------------- 编码/解码/前向 --------------------

    def encode(self, input: Tensor) -> Tensor:
        """
        input: [B, C, H, W] with H=W in {16, 32, 64}
        仅执行前 K 个下采样块，使空间到 2x2；再 flatten -> fc_z -> z
        """
        B, C, H, W = input.shape
        K = self._num_down_to_2x2(H, W)

        h = input
        for i in range(K):  # 只跑前 K 层
            h = self.enc_blocks[i](h)

        # 此时 h 应为 [B, Ck, 2, 2]
        self._enc_out_c = h.shape[1]

        flat = torch.flatten(h, start_dim=1)  # [B, Ck*4]
        z = self.fc_z(flat)                   # LazyLinear 自动推断 in_features
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        解码路径：
        z --(Linear)-> [B, Ck*4] -> [B, Ck, 2, 2]
          -- 运行 (K-1) 个反卷积块（从 dec_blocks 的合适起点开始）
          -- final_up -> 与输入同尺度
          -- conv -> 1 通道 -> Tanh
        """
        assert self._in_hw is not None and self._enc_out_c is not None, "Call forward/encode() first."
        H0, W0 = self._in_hw
        K = self._num_down_to_2x2(H0, W0)

        # 选择对应的 decoder_input（latent -> Ck*4），按 Ck 缓存
        key = str(self._enc_out_c)
        if key not in self.decoder_input_map:
            self.decoder_input_map[key] = nn.Linear(self.latent_dim, self._enc_out_c * 4).to(z.device)
        decoder_input = self.decoder_input_map[key]

        h = decoder_input(z)                      # [B, Ck*4]
        h = h.view(-1, self._enc_out_c, 2, 2)     # [B, Ck, 2, 2]

        # 计算在 dec_blocks 中的起始索引 s（让 in_channels 匹配 Ck）
        # dec_dims = reversed(hidden_dims) = [512,256,128,64,32]
        # K=5 -> Ck=512 -> s=0; K=4 -> Ck=256 -> s=1; K=3 -> Ck=128 -> s=2
        dec_dims = list(reversed(self.hidden_dims))
        Ck = self._enc_out_c
        try:
            s = dec_dims.index(Ck)
        except ValueError:
            raise RuntimeError(f"Encoded channel {Ck} not found in decoder dims {dec_dims}.")

        # 运行 (K-1) 个反卷积块：i = s ... s+(K-2)
        for i in range(s, s + (K - 1)):
            h = self.dec_blocks[i](h)

        # 第 K 次上采样到接近原始尺寸
        h = self.final_up(h)

        # 尺寸保险：若有 1 像素偏差（padding/output_padding 组合导致），插值对齐
        if h.shape[-2:] != (H0, W0):
            h = F.interpolate(h, size=(H0, W0), mode='bilinear', align_corners=False)

        h = self.final_conv(h)
        return self.final_act(h)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        返回 [reconstruction, input, z]，接口与原版保持一致
        """
        self._in_hw = input.shape[-2:]
        z = self.encode(input)
        recons = self.decode(z)
        return [recons, input, z]

    # -------------------- 损失 & SWD（与原版尽量一致） --------------------

    def loss_function(self, recons: Tensor, input: Tensor, z: Tensor) -> Tensor:
        batch_size = input.size(0)
        bias_corr = max(batch_size * (batch_size - 1), 1)  # 防止 batch=1 时除零
        reg_weight = self.reg_weight / bias_corr

        recons_loss_l2 = F.mse_loss(recons, input)
        recons_loss_l1 = F.l1_loss(recons, input)

        swd_loss = self.compute_swd(z, self.p, reg_weight)

        loss = recons_loss_l2 + recons_loss_l1 + swd_loss
        return loss

    def get_random_projections(self, latent_dim: int, num_samples: int) -> Tensor:
        if self.proj_dist == 'normal':
            rand_samples = torch.randn(num_samples, latent_dim)
        elif self.proj_dist == 'cauchy':
            rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                       torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')
        rand_proj = rand_samples / rand_samples.norm(dim=1, keepdim=True)
        return rand_proj  # [S, D]

    def compute_swd(self, z: Tensor, p: float, reg_weight: float) -> Tensor:
        """
        计算 Sliced Wasserstein Distance（投影排序差的 p-范数），
        与原版实现保持一致。
        """
        prior_z = torch.randn_like(z)  # [N, D]
        device = z.device

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0, 1).to(device)
        latent_proj = z.matmul(proj_matrix)    # [N, S]
        prior_proj  = prior_z.matmul(proj_matrix)

        w_diff = torch.sort(latent_proj.t(), dim=1)[0] - torch.sort(prior_proj.t(), dim=1)[0]
        w_dist = w_diff.pow(p)
        return reg_weight * w_dist.mean()

    def freeze_encoder(self):
        for p in self.enc_blocks.parameters():
            p.requires_grad = False
        for p in self.fc_z.parameters():
            p.requires_grad = False

