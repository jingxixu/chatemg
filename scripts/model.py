"""
References:
1) NanoGPT by Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x_1 = self.ln_1(x)
        x_1 = self.attn(x_1)
        x = x + x_1
        x_2 = self.ln_2(x)
        x_2 = self.mlp(x_2)
        x = x + x_2
        return x


class OneHot(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        x = F.one_hot(x.long(), self.output_size)
        return torch.flatten(x.float(), start_dim=-2)  # flatten as 1-D vector


class SumTokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.n_embd) for _ in range(8)]
        )

    def forward(self, x):
        # B, T, C
        x = [self.embeddings[i](x[:, :, i].long()) for i in range(8)]
        return torch.sum(torch.stack(x, dim=-1), dim=-1)


class ConcatTokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % 8 == 0
        self.embeddings = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.n_embd // 8) for _ in range(8)]
        )

    def forward(self, x):
        # B, T, C
        x = [self.embeddings[i](x[:, :, i].long()) for i in range(8)]
        return torch.cat(x, dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if config.token_embedding_type == "basic":
            self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        elif config.token_embedding_type == "FC":
            self.embedding = nn.Linear(8, config.n_embd, bias=False)
        elif (
            config.token_embedding_type == "FC_extended"
        ):  # first convert to one-hot encoding
            self.embedding = nn.Sequential(
                OneHot(config.vocab_size),
                nn.Linear(config.vocab_size * 8, config.n_embd, bias=False),
            )
        elif config.token_embedding_type == "basic_sum":
            self.embedding = SumTokenEmbedding(config)
        elif config.token_embedding_type == "basic_concat":
            self.embedding = ConcatTokenEmbedding(config)

    def forward(self, x):
        return self.embedding(x)


@dataclass
class GPTConfig:
    block_size: int = 256  # length of each block of input
    vocab_size: int = 1000  # EMG signal has 1000 possible values
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    continuous: bool = False  # True: predicting 8 means in the lm_head
    flatten: bool = False  # whether the EMG signals are flattened
    model_type: str = None
    numerical_encoding: bool = False  # whether to use numerical encoding
    token_embedding_type: str = "FC_extended"


class GPT_base(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        raise NotImplementedError

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        raise NotImplementedError


class GPT_interchannel(GPT_base):
    def __init__(self, config):
        super().__init__(config)

        self.transformer_channel = nn.ModuleDict(
            dict(
                wte=nn.Embedding(1000, config.n_embd),
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # word position embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.transformer_context = nn.ModuleDict(
            dict(
                wte=TokenEmbedding(config),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.latent_decoder = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer_channel.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer_context.wpe.weight.numel()
            n_params -= self.transformer_channel.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None, selected_channel=0):
        device = idx.device
        b, t = idx.size()[0], idx.size()[1]  # idx is now (b, t, c)
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        tok_emb_context = self.transformer_context.wte(
            idx.float()
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb_context = self.transformer_context.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)

        x_context = self.transformer_context.drop(tok_emb_context + pos_emb_context)

        for block in self.transformer_context.h:
            x_context = block(x_context)
        x_context = self.transformer_context.ln_f(x_context)

        # always select the first channel
        tok_emb_channel = self.transformer_channel.wte(
            idx[:, :, selected_channel]
        ).reshape(
            b, t, self.config.n_embd
        )  # token embeddings of shape (b, t, n_embd)
        pos_emb_channel = self.transformer_channel.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)

        x_channel = self.transformer_channel.drop(tok_emb_channel + pos_emb_channel)

        for block in self.transformer_channel.h:
            x_channel = block(x_channel)
        x_channel = self.transformer_channel.ln_f(x_channel)

        x = self.latent_decoder(torch.cat((x_channel, x_context), dim=2))

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets[:, :, selected_channel].view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        prompt_size=None,
        independent=False,
    ):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # go through all channels
            logits, _ = self(idx_cond)
            for c in range(1, 8):
                logits_c, _ = self(idx_cond.roll(shifts=-(int(c)), dims=-1))
                logits = torch.cat((logits, logits_c), dim=1)

            logits = logits.view(logits.shape[0], 1, logits.shape[1], logits.shape[2])

            logits = logits[:, -1, :] / temperature  # (b, 8, 1000)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            b, emg_c = probs.shape[:2]
            idx_next = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
            idx_next = idx_next.view(b, 1, emg_c)  # add back the time dimension

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
