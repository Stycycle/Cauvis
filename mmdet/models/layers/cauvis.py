from torch import nn
from mmdet.registry import MODELS
import torch
from torch import Tensor
from mmdet.registry import DATASETS
import numpy as np

class AuxiliaryBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims, int(dims // 16)),
            nn.Linear(int(dims // 16), dims),
            nn.SiLU(),
        )

    def fourier_transform(self, feats):
        B, L, C = feats.shape

        next_power_of_two = 1
        while next_power_of_two < L:
            next_power_of_two *= 2
        padded_feats = torch.nn.functional.pad(feats, (0, 0, 0, next_power_of_two - L))  
        fft_feats = torch.fft.fft(padded_feats, dim=1) 
        mask = torch.zeros(next_power_of_two, dtype=torch.float32, device=feats.device)
        center = next_power_of_two // 2
        mask_width = int(next_power_of_two * 0.2) // 2  
        mask_start = max(0, center - mask_width)
        mask_end = min(next_power_of_two, center + mask_width)
        mask[mask_start:mask_end] = 1.0  
        masked_fft = fft_feats * mask[None, :, None]
        ifft_feats = torch.fft.ifft(masked_fft, dim=1).real
        ifft_feats = ifft_feats[:, :L, :]

        return ifft_feats

    def forward(self, x):
        x = self.mlp(x)
        out = self.fourier_transform(x)
        return out + x


class CausalBranch(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims, dims),
            nn.SiLU()
        )

    def forward(self, x):
        return self.mlp(x)


@MODELS.register_module()
class Cauvis(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            img_size: int,
            prompt_init=None,
            query_dims: int = 256,
            token_length: int = 100,
            use_softmax: bool = True,
            link_token_to_query: bool = True,
            scale_init: float = 0.001,
            zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.token_embedding = int(img_size // patch_size) * int(img_size // patch_size)

        self.prompt = nn.Parameter(torch.zeros([self.token_length, self.embed_dims]))

        self.mlp_prompt = nn.Linear(self.embed_dims, self.embed_dims)
        self.to_out = nn.Linear(self.embed_dims, self.embed_dims)
        self.alpha = nn.Parameter(torch.tensor(self.scale_init))
        self.beta = nn.Parameter(torch.tensor(self.scale_init))
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

        self.prompt_branch = CausalBranch(self.embed_dims)
        self.aux_branch = AuxiliaryBranch(self.embed_dims)

    def cross_attention(self, x, prompt):
        attn = torch.einsum('bnc,mc->bnm', x, prompt)
        attn = attn * (self.embed_dims ** -0.5)
        attn = attn.softmax(-1)
        score = torch.einsum('bnm,mc->bnc', attn, self.mlp_prompt(prompt))
        return self.to_out(score)

    def forward(
            self, feats: Tensor,
            layer: int,
            batch_first=False,
            has_cls_token=True
    ) -> Tensor:
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)

        res_prompt = self.cross_attention(feats, self.prompt)
        main = self.prompt_branch(res_prompt) * self.alpha
        aux = self.aux_branch(res_prompt) * self.beta
        feats = feats * self.delta_scale + main + aux

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=1)
        return feats  # bnc
