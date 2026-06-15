import torch
import torch.nn.functional as F
from torch import nn


class CausalLinearAttention(nn.Module):
    """
    Implementacja Linear Attention z pracy 'Transformers are RNNs' (Katharopoulos et al., 2020).
    Złożoność: O(N) dla czasu i pamięci.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Feature map function: elu(x) + 1
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, x, state=None, use_cache=False):
        # x: [Batch, Seq_Len, Hidden]
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Aplikacja feature map phi(x)
        Q = self.feature_map(q)  # [B, L, Heads, Dim]
        K = self.feature_map(k)  # [B, L, Heads, Dim]

        if use_cache and state is not None:
            # --- TRYB REKURENCYJNY (INFERENCE) ---
            # Działa jak RNN: O(1) na krok
            # state to krotka (S, Z) z poprzedniego kroku
            # S: [B, Heads, Dim, Dim] (Numerator)
            # Z: [B, Heads, Dim]      (Denominator)
            prev_S, prev_Z = state

            # W trybie inference L=1
            # Aktualizacja stanu: S_new = S_old + (K^T * V)
            # K: [B, 1, H, D] -> [B, H, D, 1]
            # V: [B, 1, H, D] -> [B, H, 1, D]

            # Przygotowanie wymiarów pod mnożenie zewnętrzne
            k_i = K.squeeze(1)  # [B, H, D]
            v_i = v.squeeze(1)  # [B, H, D]
            q_i = Q.squeeze(1)  # [B, H, D]

            # numerator update: S += k * v^T
            new_S = prev_S + torch.einsum("bhd, bhe -> bhde", k_i, v_i)
            # denominator update: Z += k
            new_Z = prev_Z + k_i

            # Obliczenie wyjścia: V' = (Q * S) / (Q * Z)
            num = torch.einsum("bhd, bhde -> bhe", q_i, new_S)
            den = torch.einsum("bhd, bhd -> bh", q_i, new_Z) + 1e-6  # Epsilon dla stabilności

            output = num / den.unsqueeze(-1)  # [B, H, D]
            output = output.unsqueeze(1)  # [B, 1, H, D]

            return self.out_proj(output.reshape(B, L, H)), (new_S, new_Z)

        else:
            # --- TRYB RÓWNOLEGŁY (TRENING) ---
            # Używamy sumy kumulacyjnej (cumsum) zamiast pętli for

            # 1. Obliczamy iloczyn zewnętrzny K i V dla każdego kroku
            # KV = einsum(K, V) -> [B, L, Heads, Dim_K, Dim_V]
            KV = torch.einsum("blhd,blhe->blhde", K, v)

            # 2. Suma kumulacyjna wzdłuż wymiaru czasu (L) - to jest "maska kauzalna"
            # S_i = suma_{j=0..i} (K_j * V_j^T)
            cum_KV = torch.cumsum(KV, dim=1)

            # 3. Suma kumulacyjna samych kluczy dla mianownika
            # Z_i = suma_{j=0..i} K_j
            cum_K = torch.cumsum(K, dim=1)

            # 4. Obliczenie atencji
            # Licznik: Q_i * S_i
            numerator = torch.einsum("blhd,blhde->blhe", Q, cum_KV)

            # Mianownik: Q_i * Z_i
            denominator = torch.einsum("blhd,blhd->blh", Q, cum_K) + 1e-6

            attn_output = numerator / denominator.unsqueeze(-1)

            # Scalanie głów i projekcja
            attn_output = attn_output.reshape(B, L, H)
            return self.out_proj(attn_output), None


# Prosty Feed Forward (MLP)
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LinearTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_ff_size, dropout):
        super().__init__()
        self.attn = CausalLinearAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size, intermediate_ff_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None, use_cache=False):
        # Residual connection + Norm
        attn_out, new_state = self.attn(self.ln1(x), state, use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x, new_state
