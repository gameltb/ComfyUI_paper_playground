import dataclasses
import math
from typing import Optional, Self, Tuple

import nunchaku._C  # noqa: F401
import torch
from diffusers.models.normalization import RMSNorm
from torch import nn


class FluxModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transformer_blocks = nn.ModuleList([JointTransformerBlock(3072, 24, 3072, False) for _ in range(19)])
        self.single_transformer_blocks = nn.ModuleList(
            [FluxSingleTransformerBlock(3072, 24, 3072, 4) for _ in range(38)]
        )


def gemv_awq(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    M,
    n,
    k,
    group_size,
):
    DTYPE = torch.bfloat16
    assert in_feats.dtype == DTYPE
    assert scaling_factors.dtype == DTYPE
    assert zeros.dtype == DTYPE
    assert in_feats.is_contiguous()
    assert kernel.is_contiguous()
    assert scaling_factors.is_contiguous()
    assert zeros.is_contiguous()
    outshape = list(in_feats.shape)
    outshape[-1] = n
    out = torch.empty(outshape, device=in_feats.device, dtype=DTYPE, memory_format=torch.contiguous_format)
    torch.ops.nunchaku.gemv_awq.default(in_feats, kernel, scaling_factors, zeros, M, n, k, group_size, out)
    return out


class GEMV_AWQ(nn.Module):
    def __init__(
        self, in_features, out_features, bias: bool = True, lora: bool = False, device=None, dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 64

        self.qweight = nn.Parameter(
            torch.empty(
                (int(out_features / 4), math.ceil(in_features / 8) * 4),
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )
        self.wscales = nn.Parameter(
            torch.empty((math.ceil(in_features / self.group_size), out_features), **factory_kwargs)
        )
        self.wzeros = nn.Parameter(
            torch.empty((math.ceil(in_features / self.group_size), out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.bias = None

        self.lora_rank = 0
        self.lora_scale = 1.0

        if lora:
            self.lora_down = nn.Parameter(torch.empty((self.lora_rank, in_features), **factory_kwargs))
            self.lora_up = nn.Parameter(torch.empty((out_features, self.lora_rank), **factory_kwargs))

    def forward(
        self,
        x: torch.Tensor,
    ):
        M = int(x.numel() / x.shape[-1])
        out = gemv_awq(
            x,
            self.qweight,
            self.wscales,
            self.wzeros,
            M,
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            assert out.numel() == self.bias.numel()
            out = out + self.bias.view(out.shape)

        if self.lora_rank > 0:
            raise Exception()
        return out


@dataclasses.dataclass
class QuantizedActivation:
    act: torch.Tensor
    ascales: torch.Tensor
    lora_act: torch.Tensor
    is_unsigned = False


class GEMM_W4A4(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = 64

        self.qweight = nn.Parameter(
            torch.empty(
                (out_features, int(in_features / 2)),
                dtype=torch.int8,
                device=device,
            ),
            requires_grad=False,
        )
        self.wscales = nn.Parameter(torch.empty((int(in_features / self.group_size), out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))

        self.lora_rank = 32
        self.lora_scale = 1.0
        self.lora_down = nn.Parameter(torch.empty((in_features, self.lora_rank), **factory_kwargs))
        self.lora_up = nn.Parameter(torch.empty((out_features, self.lora_rank), **factory_kwargs))

        self.smooth = nn.Parameter(torch.empty((in_features,), **factory_kwargs))

        self.lora_scales = [1.0 for _ in range(math.ceil(self.lora_rank / 16))]

    def quantize(
        self,
        x: torch.Tensor,
    ):
        assert x.is_contiguous()
        M = int(x.numel() / x.shape[-1])
        shape = list(x.shape)
        shape[-1] = int(self.in_features / 2)
        qact = QuantizedActivation(
            act=torch.empty(shape, dtype=torch.int8, device=self.qweight.device),
            ascales=torch.empty((int(self.in_features / 64), M), dtype=self.wscales.dtype, device=self.qweight.device),
            lora_act=torch.empty((M, self.lora_rank), dtype=torch.float32, device=self.qweight.device),
        )

        torch.ops.nunchaku.quantize_w4a4_act_fuse_lora.default(
            x, qact.act, qact.ascales, self.lora_down, qact.lora_act, self.smooth
        )
        return qact

    def forward_quant(self, x, nextGEMM: Self = None):
        qact = self.quantize(x)
        return self._forward_quant(qact, nextGEMM)

    def _forward_quant(self, qact: QuantizedActivation, nextGEMM: Self = None):
        M = int(qact.act.numel() / qact.act.shape[-1])
        out = None
        qout = QuantizedActivation(None, None, None)
        next_lora = None
        next_smooth = None
        M = int(qact.act.numel() / qact.act.shape[-1])

        if nextGEMM is None:
            shape = list(qact.act.shape)
            shape[-1] = self.out_features
            out = torch.empty(shape, dtype=torch.bfloat16, device=self.qweight.device)
        else:
            shape = list(qact.act.shape)
            shape[-1] = int(self.out_features / 2)
            qout = QuantizedActivation(
                act=torch.empty(shape, dtype=torch.int8, device=self.qweight.device),
                ascales=torch.empty(
                    (int(self.out_features / 64), M), dtype=self.wscales.dtype, device=self.qweight.device
                ),
                lora_act=torch.empty((M, self.lora_rank), dtype=torch.float32, device=self.qweight.device),
            )
            qout.is_unsigned = True
            next_lora = nextGEMM.lora_down
            next_smooth = nextGEMM.smooth
        torch.ops.nunchaku.gemm_w4a4.default(
            qact.act,
            self.qweight,
            out,
            qout.act,
            qact.ascales,
            self.wscales,
            qout.ascales,
            None,
            qact.lora_act,
            self.lora_up,
            next_lora,
            qout.lora_act,
            None,
            None,
            None,
            self.bias,
            next_smooth,
            qact.is_unsigned,
            self.lora_scales,
        )
        if out is not None:
            return out
        return qout

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        pool: Optional[torch.Tensor],
        norm_q: torch.Tensor,
        norm_k: torch.Tensor,
        rotary_emb: torch.Tensor,
    ):
        qact = self.quantize(x)
        assert rotary_emb.is_contiguous()
        torch.ops.nunchaku.gemm_w4a4.default(
            qact.act,
            self.qweight,
            out,
            None,
            qact.ascales,
            self.wscales,
            None,
            pool,
            qact.lora_act,
            self.lora_up,
            None,
            None,
            norm_q,
            norm_k,
            rotary_emb,
            self.bias,
            None,
            qact.is_unsigned,
            self.lora_scales,
        )
        return out


class AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        GEMM = GEMV_AWQ
        self.linear = GEMM(dim, 3 * dim, True)
        self.norm = nn.LayerNorm(dim, 1e-6, False)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        # shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        shift_msa, scale_msa, gate_msa = emb.reshape((1, -1, 3)).unbind(2)
        # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = self.norm(x) * (scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim, pre_only) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        GEMM = GEMV_AWQ
        self.linear = GEMM(dim, 2 * dim if pre_only else 6 * dim, True)
        self.norm = nn.LayerNorm(dim, 1e-6, False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_linear = self.linear(self.silu(emb))
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_linear.chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_linear.reshape((1, -1, 6)).unbind(2)
        # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = self.norm(x) * scale_msa + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class JointTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only) -> None:
        super().__init__()
        self.dim = dim
        self.dim_head = int(attention_head_dim / num_attention_heads)
        self.num_heads = num_attention_heads
        self.context_pre_only = context_pre_only

        GEMM = GEMM_W4A4
        self.norm1 = AdaLayerNormZero(self.dim, False)
        self.norm1_context = AdaLayerNormZero(self.dim, self.context_pre_only)

        self.qkv_proj = GEMM(dim, 3 * dim, True)
        self.qkv_proj_context = GEMM(dim, 3 * dim, True)

        self.norm_k = RMSNorm(self.dim_head, 1e-6)
        self.norm_q = RMSNorm(self.dim_head, 1e-6)

        self.norm_added_k = RMSNorm(self.dim_head, 1e-6)
        self.norm_added_q = RMSNorm(self.dim_head, 1e-6)

        # self.attn = Attention(num_attention_heads, attention_head_dim / num_attention_heads)

        self.out_proj = GEMM(dim, dim, True)
        self.out_proj_context = GEMM(dim, dim, True)

        self.norm2 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)

        self.mlp_fc1 = GEMM(dim, dim * 4, True)
        self.mlp_fc2 = GEMM(dim * 4, dim, True)

        self.mlp_context_fc1 = GEMM(dim, dim * 4, True)
        self.mlp_context_fc2 = GEMM(dim * 4, dim, True)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        assert image_rotary_emb.shape[2] == batch_size * (txt_tokens + img_tokens)
        # [bs, tokens, head_dim / 2, 1, 2] (sincos)
        image_rotary_emb = image_rotary_emb.reshape([batch_size, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...].contiguous()  # .to(self.dtype)
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...].contiguous()  # .to(self.dtype)

        #####################
        rotary_emb = rotary_emb_img
        rotary_emb_context = rotary_emb_txt

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        num_tokens_img = hidden_states.shape[1]
        num_tokens_context = encoder_hidden_states.shape[1]
        # POOL_SIZE = 128
        # poolTokens = num_tokens_img / POOL_SIZE + num_tokens_context / POOL_SIZE
        concat = torch.empty(
            (batch_size, num_tokens_img + num_tokens_context, self.dim * 3),
            dtype=norm_hidden_states.dtype,
            device=norm_hidden_states.device,
        )
        for i in range(batch_size):
            qkv = concat[i : i + 1, :num_tokens_img]
            qkv_context = concat[i : i + 1, num_tokens_img : num_tokens_img + num_tokens_context]
            pool_qkv = None
            pool_qkv_context = None

            self.qkv_proj(
                norm_hidden_states[i : i + 1], qkv, pool_qkv, self.norm_q.weight, self.norm_k.weight, rotary_emb
            )
            self.qkv_proj_context(
                norm_encoder_hidden_states[i : i + 1],
                qkv_context,
                pool_qkv_context,
                self.norm_added_q.weight,
                self.norm_added_k.weight,
                rotary_emb_context,
            )

        num_heads = self.num_heads
        dim_head = self.dim_head
        num_tokens = concat.shape[1]
        raw_attn_output = nunchaku_attention(concat, batch_size, num_tokens, num_heads, dim_head)
        raw_attn_output = raw_attn_output.view([batch_size, num_tokens_img + num_tokens_context, num_heads, dim_head])

        raw_attn_output_split = raw_attn_output[:, :num_tokens_img].reshape(
            [batch_size, num_tokens_img, num_heads * dim_head]
        )
        context_raw_attn_output_split = raw_attn_output[
            :, num_tokens_img : num_tokens_img + num_tokens_context
        ].reshape([batch_size, num_tokens_context, num_heads * dim_head])
        attn_output = self.out_proj.forward_quant(raw_attn_output_split)
        context_attn_output = self.out_proj_context.forward_quant(context_raw_attn_output_split)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        # norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_hidden_states = norm_hidden_states * scale_mlp + shift_mlp

        # ff_output = self.ff(norm_hidden_states)
        ff_output = self.mlp_fc2._forward_quant(self.mlp_fc1.forward_quant(norm_hidden_states, self.mlp_fc2))
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        # norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * c_scale_mlp + c_shift_mlp

        # context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = self.mlp_context_fc2._forward_quant(
            self.mlp_context_fc1.forward_quant(norm_encoder_hidden_states, self.mlp_context_fc2)
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio) -> None:
        super().__init__()
        self.dim = dim
        self.dim_head = int(attention_head_dim / num_attention_heads)
        self.num_heads = num_attention_heads
        self.mlp_hidden_dim = dim * mlp_ratio

        GEMM = GEMM_W4A4

        self.norm = AdaLayerNormZeroSingle(self.dim)

        self.mlp_fc1 = GEMM(dim, self.mlp_hidden_dim, True)
        self.mlp_fc2 = GEMM(self.mlp_hidden_dim, dim, True)

        self.qkv_proj = GEMM(dim, dim * 3, True)

        self.norm_q = RMSNorm(self.dim_head, 1e-6)
        self.norm_k = RMSNorm(self.dim_head, 1e-6)

        # self.attn = Attention(num_attention_heads, attention_head_dim / num_attention_heads)

        self.out_proj = GEMM(dim, dim, True)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states.clone()
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        batch_size = hidden_states.shape[0]
        num_tokens = hidden_states.shape[1]
        qkv = torch.empty(
            (batch_size, num_tokens, self.dim * 3),
            dtype=norm_hidden_states.dtype,
            device=norm_hidden_states.device,
        )
        rotary_emb = image_rotary_emb
        self.qkv_proj(norm_hidden_states, qkv, None, self.norm_q.weight, self.norm_k.weight, rotary_emb)

        num_heads = self.num_heads
        dim_head = self.dim_head
        num_tokens = qkv.shape[1]
        raw_attn_output = nunchaku_attention(qkv, batch_size, num_tokens, num_heads, dim_head)
        raw_attn_output = raw_attn_output.reshape((batch_size, num_tokens, num_heads * dim_head))
        attn_output = self.out_proj.forward_quant(raw_attn_output)

        mlp_hidden_states = self.mlp_fc2._forward_quant(self.mlp_fc1.forward_quant(norm_hidden_states, self.mlp_fc2))

        # mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        # joint_attention_kwargs = joint_attention_kwargs or {}
        # attn_output = self.attn(
        #     hidden_states=norm_hidden_states,
        #     image_rotary_emb=image_rotary_emb,
        #     **joint_attention_kwargs,
        # )
        hidden_states = attn_output + mlp_hidden_states
        # gate = gate.unsqueeze(1)
        hidden_states = gate * hidden_states
        # hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        # gate = gate.unsqueeze(1)
        # hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


def nunchaku_attention(qkv, batch_size, num_tokens, num_heads, dim_head):
    reshaped = qkv.view([batch_size * num_tokens, num_heads * 3, dim_head])
    q = reshaped[:, 0:num_heads]
    k = reshaped[:, num_heads : num_heads * 2]
    v = reshaped[:, num_heads * 2 : num_heads * 3]
    # raw_attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    from block_sparse_attn.block_sparse_attn_interface import _block_sparse_attn_forward

    POOL_SIZE = 128
    pool_tokens = int(num_tokens / POOL_SIZE)
    pool_score = torch.zeros((batch_size, num_heads, pool_tokens, pool_tokens), dtype=torch.int32, device=q.device)
    blockmask = torch.topk(pool_score, pool_tokens)[1].to(dtype=torch.int32, device=q.device)
    cu_seqlens = torch.Tensor([i * num_tokens for i in range(batch_size + 1)]).to(dtype=torch.int32, device=q.device)
    headmask_type = torch.Tensor([i + 1 for i in range(num_heads)]).to(dtype=torch.int32, device=q.device)
    raw_attn_output = _block_sparse_attn_forward(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        POOL_SIZE,
        POOL_SIZE,
        headmask_type,
        None,
        blockmask,
        num_tokens,
        num_tokens,
        0.0,
        q.shape[-1] ** (-0.5),
        False,
        False,
        False,
        -1,
        -1,
    )

    # reshaped = qkv.view([batch_size , num_tokens, num_heads * 3, dim_head])
    # q = reshaped[:,:, 0:num_heads]
    # k = reshaped[:,:, num_heads : num_heads * 2]
    # v = reshaped[:,:, num_heads * 2 : num_heads * 3]
    # raw_attn_output = attention_blocksparse_ref(q, k, v, None,None,None)
    return raw_attn_output[0]


from einops import rearrange, repeat


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_blocksparse_ref(
    q,
    k,
    v,
    mixed_mask,
    m_block_dim,
    n_block_dim,
    query_padding_mask=None,
    key_padding_mask=None,
    p_dropout=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
):
    # q, k, v = qkv.float().unbind(dim=2)
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    # local mask
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))

    # scores.masked_fill_(rearrange(mixed_mask, "b h t s -> b h t s"), float("-inf"))

    # print("processed blockmask: ", rearrange(~base_blockmask, "h t s -> 1 h t s"))

    attention = torch.softmax(scores, dim=-1).to(v.dtype)

    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(torch.bitwise_or(local_mask, rearrange(mixed_mask, "b h t s -> b h t s")), dim=-1, keepdim=True),
            0.0,
        )

    # attention = attention.masked_fill(rearrange(mixed_mask, "b h t s -> b h t s"), 0.0)

    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - p_dropout)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


if __name__ in ("__main__", "<run_path>"):
    import accelerate
    import safetensors.torch

    with torch.inference_mode():
        with accelerate.init_empty_weights():
            flux_model = FluxModel()
        state_dict = safetensors.torch.load_file("svdq-int4-flux.1-dev.safetensors")
        flux_model.load_state_dict(state_dict=state_dict, assign=True)

        jb1 = flux_model.transformer_blocks[0].cuda()
        sb1 = flux_model.single_transformer_blocks[0].cuda()

        # Batch = 1
        # Width = 64
        # Height = 64
        # dim = 3072
        # Token = 256
        # hidden_states = torch.randn([Batch, Width * Height, dim], device="cuda", dtype=torch.bfloat16)
        # encoder_hidden_states = torch.randn([Batch, Token, dim], device="cuda", dtype=torch.bfloat16)
        # temb = torch.randn([Batch, Token, dim], device="cuda", dtype=torch.bfloat16)

        jb1_args = safetensors.torch.load_file("flux_DoubleStreamBlock_0.st")
        hidden_states = jb1_args["img_0"].cuda()
        encoder_hidden_states = jb1_args["txt_0"].cuda()
        temb = jb1_args["vec_0"].cuda()
        image_rotary_emb = jb1_args["pe_0"].cuda()[:, :, :, :, :1]
        jb1(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # sb1(hidden_states, temb, image_rotary_emb)
        print(torch.cuda.memory_summary())
