import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_

import torch.utils.checkpoint as checkpoint
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


# BEiTv2 block
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x_attn, attn_weights = self.attn(
                xn,
                xn,
                xn,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x_attn, attn_weights = self.attn(
                xn,
                xn,
                xn,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x, attn_weights


class Attention_rel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.proj_q = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_k = nn.Linear(dim, all_head_dim, bias=False)
        self.proj_v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, rel_pos_bias=None, key_padding_mask=None):
        # rel_pos_bias: B L L C/h
        # key_padding_mask - float with -inf
        B, N, C = q.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #    qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = F.linear(input=q, weight=self.proj_q.weight, bias=self.q_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = F.linear(input=k, weight=self.proj_k.weight, bias=None)
        k = k.reshape(B, k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        v = F.linear(input=v, weight=self.proj_v.weight, bias=self.v_bias)
        v = v.reshape(B, v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if rel_pos_bias is not None:
            bias = torch.einsum("bhic,bijc->bhij", q, rel_pos_bias).type_as(attn)
            attn = attn + bias
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float16
            ), "incorrect mask dtype"
            bias = torch.min(
                key_padding_mask[:, None, :], key_padding_mask[:, :, None]
            ).type_as(attn)
            bias[
                torch.max(key_padding_mask[:, None, :], key_padding_mask[:, :, None])
                < 0
            ] = 0
            attn = attn + bias.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        if rel_pos_bias is not None:
            x = x + torch.einsum("bhij,bijc->bihc", attn, rel_pos_bias)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


# BEiTv2 block
class Block_rel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_rel(
            dim, num_heads, attn_drop=attn_drop, qkv_bias=qkv_bias
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values is not None:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, key_padding_mask=None, rel_pos_bias=None, kv=None):
        if self.gamma_1 is None:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x_attn, attn_weights = self.attn(
                xn,
                kv,
                kv,
                rel_pos_bias=rel_pos_bias,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            kv = xn if kv is None else self.norm1(kv)
            x_attn, attn_weights = self.attn(
                xn,
                kv,
                kv,
                rel_pos_bias=rel_pos_bias,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.drop_path(self.gamma_1 * self.drop_path(x_attn))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn_weights


class LocalBlock(nn.Module):
    def __init__(
        self,
        dim=192,
        num_heads=192 // 64,
        mlp_ratio=4,
        drop_path=0,
        init_values=1,
        **kwargs,
    ):
        super().__init__()
        self.proj_rel_bias = nn.Linear(dim // num_heads, dim // num_heads)
        self.block = Block_rel(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            init_values=init_values,
        )

    def forward(self, x, nbs, key_padding_mask=None, rel_pos_bias=None):
        B, Lmax, C = x.shape
        mask = (
            key_padding_mask
            if not (key_padding_mask is None)
            else torch.ones(B, Lmax, dtype=torch.bool)
        )

        m = torch.gather(mask.unsqueeze(1).expand(-1, Lmax, -1), 2, nbs)
        attn_mask = torch.zeros(m.shape)
        attn_mask[~mask] = -torch.inf
        attn_mask = attn_mask[mask]

        if rel_pos_bias is not None:
            rel_pos_bias = torch.gather(
                rel_pos_bias,
                2,
                nbs.unsqueeze(-1).expand(-1, -1, -1, rel_pos_bias.shape[-1]),
            )
            rel_pos_bias = rel_pos_bias[mask]
            rel_pos_bias = self.proj_rel_bias(rel_pos_bias).unsqueeze(1)

        xl = torch.gather(
            x.unsqueeze(1).expand(-1, Lmax, -1, -1),
            2,
            nbs.unsqueeze(-1).expand(-1, -1, -1, C),
        )
        xl = xl[mask]
        # modify only the node (0th element)
        xl = self.block(
            xl[:, :1],
            rel_pos_bias=rel_pos_bias,
            key_padding_mask=attn_mask[:, :1],
            kv=xl,
        )
        x = torch.zeros(x.shape, dtype=xl.dtype)
        x[mask] = xl.squeeze(1)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim) * (-emb)).type_as(x[..., None])
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Extractor(nn.Module):
    def __init__(self, dim_base=128, dim=384):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        self.combined_dom_type_emb = nn.Embedding(5, dim_base // 2)
        self.emb2 = SinusoidalPosEmb(dim=dim_base // 2)
        self.proj = nn.Sequential(
            nn.Linear(6 * dim_base, 6 * dim_base),
            nn.LayerNorm(6 * dim_base),
            nn.GELU(),
            nn.Linear(6 * dim_base, dim),
        )

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        charge = x["charge"] if Lmax is None else x["charge"][:, :Lmax]
        time = x["dom_time"] if Lmax is None else x["dom_time"][:, :Lmax]

        combined_dom_type = (
            x["combined_dom_type"] if Lmax is None else x["combined_dom_type"][:, :Lmax]
        )
        combined_dom_type.to(dtype=pos.dtype)
        length = torch.log10(x["L0"].to(dtype=pos.dtype))

        x = torch.cat(
            [
                self.emb(4096 * pos).flatten(-2),
                self.emb(1024 * charge),
                self.emb(4096 * time),
                self.combined_dom_type_emb(combined_dom_type),
                self.emb2(length).unsqueeze(1).expand(-1, pos.shape[1], -1),
            ],
            -1,
        )
        x = self.proj(x)
        return x


class Rel_ds(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, Lmax=None):
        pos = x["pos"] if Lmax is None else x["pos"][:, :Lmax]
        time = x["dom_time"] if Lmax is None else x["dom_time"][:, :Lmax]
        ds2 = (pos[:, :, None] - pos[:, None, :]).pow(2).sum(-1) - (
            (time[:, :, None] - time[:, None, :]) * (3e4 / 500 * 3e-1)
        ).pow(2)
        d = torch.sign(ds2) * torch.sqrt(torch.abs(ds2))
        emb = self.emb(1024 * d.clip(-4, 4))
        rel_attn = self.proj(emb)
        return rel_attn, emb


class DeepIceModel(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=12,
        use_checkpoint=False,
        head_size=32,
        depth_rel=4,
        n_rel=1,
        dim_out=1,
        **kwargs,
    ):
        super().__init__()
        self.extractor = Extractor(dim_base, dim)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [Block_rel(dim=dim, num_heads=dim // head_size) for i in range(depth_rel)]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        self.proj_out = nn.Linear(dim, dim_out)
        self.out_act = nn.Softplus()  # nn.ReLU()
        self.use_checkpoint = use_checkpoint
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)
        self.n_rel = n_rel

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    # def forward(self, x0, return_attention_weights=False):
    #     mask = x0["mask"]
    #     Lmax = mask.sum(-1).max()
    #     x = self.extractor(x0, Lmax)
    #     rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)

    #     mask = mask[:, :Lmax]
    #     B, _ = mask.shape

    #     attn_mask = torch.zeros(mask.shape).type_as(rel_pos_bias)
    #     attn_mask[~mask] = -torch.inf

    #     if return_attention_weights:
    #         attn_weight_rel_list = []
    #         for i, blk in enumerate(self.sandwich):
    #             x, attn_weight_rel = blk(x, attn_mask, rel_pos_bias)
    #             attn_weight_rel_list.append(attn_weight_rel)
    #             if i + 1 == self.n_rel:
    #                 rel_pos_bias = None
    #         attn_weight_rel_tensor = torch.stack(attn_weight_rel_list, dim=1)
    #     else:
    #         for i, blk in enumerate(self.sandwich):
    #             x, _ = blk(x, attn_mask, rel_pos_bias)
    #             if i + 1 == self.n_rel:
    #                 rel_pos_bias = None

    #     mask = torch.cat(
    #         [
    #             torch.ones(
    #                 B,
    #                 1,
    #             ).type_as(mask),
    #             mask,
    #         ],
    #         1,
    #     )

    #     cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)

    #     attn_mask = torch.zeros(mask.shape, dtype=mask.dtype).type_as(cls_token)
    #     attn_mask[~mask] = -torch.inf

    #     x = torch.cat([cls_token, x], 1)

    #     if return_attention_weights:
    #         attn_weight_list = []
    #         for blk in self.blocks:
    #             if self.use_checkpoint:
    #                 x, attn_weight = checkpoint.checkpoint(blk, x, None, attn_mask)
    #                 attn_weight_list.append(attn_weight)
    #             else:
    #                 x, attn_weight = blk(x, None, attn_mask)
    #                 attn_weight_list.append(attn_weight)
    #         attn_weight_tensor = torch.stack(attn_weight_list, dim=1)
    #     else:
    #         for blk in self.blocks:
    #             if self.use_checkpoint:
    #                 x, _ = checkpoint.checkpoint(blk, x, None, attn_mask)
    #             else:
    #                 x, _ = blk(x, None, attn_mask)

    #     x = self.proj_out(x[:, 0])  # cls token
    #     x = self.out_act(x)

    #     if return_attention_weights:
    #         return {
    #             "pred": x.squeeze(1),
    #             "mask": attn_mask,
    #             "attn_weights": attn_weight_tensor,
    #             "rel_attn_weights": attn_weight_rel_tensor,
    #         }
    #     else:
    #         return {
    #             "pred": x.squeeze(1),
    #         }

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)

        mask = mask[:, :Lmax]
        B, _ = mask.shape

        attn_mask = torch.zeros(mask.shape).type_as(rel_pos_bias)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x, _ = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [
                torch.ones(
                    B,
                    1,
                ).type_as(mask),
                mask,
            ],
            1,
        )

        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)

        attn_mask = torch.zeros(mask.shape, dtype=mask.dtype).type_as(cls_token)
        attn_mask[~mask] = -torch.inf

        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x, _ = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x, _ = blk(x, None, attn_mask)

        x = self.proj_out(x[:, 0])  # cls token
        x = self.out_act(x)
        return x.squeeze(1)

    def get_attn_weights(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)

        mask = mask[:, :Lmax]
        B, _ = mask.shape

        attn_mask = torch.zeros(mask.shape).type_as(rel_pos_bias)
        attn_mask[~mask] = -torch.inf

        attn_weight_rel_list = []
        for i, blk in enumerate(self.sandwich):
            x, attn_weight_rel = blk(x, attn_mask, rel_pos_bias)
            attn_weight_rel_list.append(attn_weight_rel)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        attn_weight_rel_tensor = torch.stack(attn_weight_rel_list, dim=1)

        mask = torch.cat(
            [
                torch.ones(
                    B,
                    1,
                ).type_as(mask),
                mask,
            ],
            1,
        )

        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)

        attn_mask = torch.zeros(mask.shape, dtype=mask.dtype).type_as(cls_token)
        attn_mask[~mask] = -torch.inf

        x = torch.cat([cls_token, x], 1)

        attn_weight_list = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x, attn_weight = checkpoint.checkpoint(blk, x, None, attn_mask)
                attn_weight_list.append(attn_weight)
            else:
                x, attn_weight = blk(x, None, attn_mask)
                attn_weight_list.append(attn_weight)

        attn_weight_tensor = torch.stack(attn_weight_list, dim=1)

        x = self.proj_out(x[:, 0])  # cls token
        x = self.out_act(x)
        return {
            "mask": attn_mask,
            "attn_weights": attn_weight_tensor,
            "rel_attn_weights": attn_weight_rel_tensor,
        }

    # def forward(self, x0):
    #     mask = x0["mask"]
    #     Lmax = mask.sum(-1).max()
    #     x = self.extractor(x0, Lmax)
    #     rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
    #     # nbs = get_nbs(x0, Lmax)
    #     mask = mask[:, :Lmax]
    #     B, _ = mask.shape

    #     attn_mask = torch.zeros(mask.shape).type_as(
    #         rel_pos_bias
    #     )
    #     attn_mask[~mask] = -torch.inf

    #     for i, blk in enumerate(self.sandwich):
    #         # if isinstance(blk, LocalBlock):
    #         #     x = blk(x, nbs, mask, rel_enc)
    #         # else:
    #         x = blk(x, attn_mask, rel_pos_bias)
    #         if i + 1 == self.n_rel:
    #             rel_pos_bias = None

    #     mask = torch.cat(
    #         [
    #             torch.ones(
    #                 B,
    #                 1,
    #             ).type_as(mask),
    #             mask,
    #         ],
    #         1,
    #     )
    #     cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)

    #     attn_mask = torch.zeros(mask.shape, dtype=mask.dtype).type_as(cls_token)
    #     attn_mask[~mask] = -torch.inf

    #     x = torch.cat([cls_token, x], 1)

    #     for blk in self.blocks:
    #         if self.use_checkpoint:
    #             x = checkpoint.checkpoint(blk, x, None, attn_mask)
    #         else:
    #             x = blk(x, None, attn_mask)

    #     x = self.proj_out(x[:, 0])  # cls token
    #     x = self.out_act(x)
    #     return x.squeeze(1)
