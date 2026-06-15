import torch
from torch import nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


class CrossAttention(nn.Module):
    """
    Standardowa atencja krzyżowa:
    - Query pochodzi z Dekodera (treść muzyczna)
    - Key/Value pochodzi z Enkodera (latent Z)
    """

    def __init__(self, config: GPTNeoXConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states (Query): [Batch, Seq_Len, Hidden]
        # encoder_hidden_states (Key/Value): [Batch, 1, Hidden] (nasze Z)

        B, L, _ = hidden_states.shape
        S = encoder_hidden_states.shape[1]  # Długość sekwencji Z (zazwyczaj 1)

        # 1. Projekcje
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        # 2. Rozdzielenie na głowy [B, Seq, Heads, Dim] -> [B, Heads, Seq, Dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Attention Score (Q * K^T)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 4. Agregacja (Weights * V)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

        return self.out_proj(attn_output)
