import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np
from algorithms.utils import SquashedMultivariateNormalDiag, DistributionalValueHead
from models.utils.base import Torso

def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    """
    Sinusoidal positional encoding 1D.
    Ref: https://arxiv.org/abs/1706.03762
    """
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        if cond_fn is not None:
            # Placeholder for Adaptive LayerNorm logic if needed later
            x = cond_fn(x)
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, dim_feedforward, dropout=dropout)

    def forward(self, x, mask=None, cond_fn=None):
        # Pre-norm architecture
        x2 = self.ln1(x)
        if cond_fn is not None:
            x2 = cond_fn(x2)
            
        attn_out, _ = self.attn(x2, x2, x2, attn_mask=mask)
        x = x + attn_out
        
        x = x + self.ff(self.ln2(x), cond_fn=cond_fn)
        return x

class CausalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, cond_fns=None):
        seq_len = x.size(1)
        # Create causal mask for nn.MultiheadAttention: (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        cond_fns = iter(cond_fns) if cond_fns is not None else iter([None]*len(self.layers))
        
        for layer in self.layers:
            x = layer(x, mask, cond_fn=next(cond_fns))
        return x

class TransformerTorso(Torso):
    def __init__(self, observation_space, d_model, nhead, num_layers, dim_feedforward, observation_normalizer=None, history_size=0, device=torch.device("cpu")):
        super().__init__(observation_normalizer)
        self.observation_size = observation_space.shape[0]
        self.history_size = history_size
        self.d_model = d_model
        self.device = device
        
        if history_size > 0:
            self.input_dim = self.observation_size // self.history_size
        else:
            self.input_dim = self.observation_size
            
        # Linear projection to d_model
        self.input_proj = nn.Linear(self.input_dim, d_model)
        
        # Causal Transformer
        self.transformer = CausalTransformer(d_model, nhead, num_layers, dim_feedforward)
        
        self.output_size = d_model
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def build_token_sequence(self, observations):
        batch_size = observations.size(0)
        if self.history_size > 0:
            # observations: (batch, history_size * input_dim)
            # Reshape to (batch, history_size, input_dim)
            tokens = observations.view(batch_size, self.history_size, self.input_dim)
            return tokens
        return observations.unsqueeze(1)
    
    def forward(self, observations):
        observations = self.normalize(observations)
        
        # Build token sequence
        x = self.build_token_sequence(observations)
        seq_len = x.size(1)
            
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional embeddings
        pos_emb = posemb_sincos_1d(seq_len, self.d_model, device=x.device, dtype=x.dtype)
        x = x + pos_emb.unsqueeze(0) # (1, seq_len, d_model)
        
        # Apply Transformer
        x = self.transformer(x)
        
        # Return last token output
        return x[:, -1, :]

class TransformerActor(nn.Module):
    def __init__(self, observation_space, action_space, d_model, nhead, num_layers, dim_feedforward, observation_normalizer=None, head_type="gaussian", history_size=0, device=torch.device("cpu")):
        super().__init__()
        self.torso = TransformerTorso(observation_space, d_model, nhead, num_layers, dim_feedforward, observation_normalizer, history_size, device)
        self.head_type = head_type
        self.action_size = action_space.shape[0]
        self.device = device

        if head_type == "gaussian":
            self.mean_layer = nn.Sequential(nn.Linear(d_model, self.action_size), nn.Tanh())
            self.std_layer = nn.Sequential(nn.Linear(d_model, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "gaussian_multivariate":
            self.mean_layer = nn.Sequential(nn.Linear(d_model, self.action_size), nn.Identity())
            self.std_layer = nn.Sequential(nn.Linear(d_model, self.action_size), nn.Softplus())
            self.std_min, self.std_max = 1e-4, 1
        elif head_type == "deterministic":
            self.action_layer = nn.Sequential(nn.Linear(d_model, self.action_size), nn.Tanh())
        
        self.to(device)

        # Print parameter count
        print(f"Parameter count actor: {sum(p.numel() for p in self.parameters())}")

    def forward(self, observations):
        out = self.torso(observations)
        out = F.gelu(out)
        
        if self.head_type == "deterministic":
            return self.action_layer(out)
        elif self.head_type == "gaussian":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return Normal(mean, std)
        elif self.head_type == "gaussian_multivariate":
            mean = self.mean_layer(out)
            std = torch.clamp(self.std_layer(out), self.std_min, self.std_max)
            return SquashedMultivariateNormalDiag(mean, std)

    def get_action(self, observations):
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).to(self.device).float()
        
        out = self.forward(observations)
        if hasattr(out, 'mean'):
             return out.mean
        if hasattr(out, 'loc'):
            return out.loc
        return out

class TransformerCritic(nn.Module):
    def __init__(self, observation_space, action_space, d_model, nhead, num_layers, dim_feedforward, observation_normalizer=None, history_size=0, critic_type="deterministic", device=torch.device("cpu")):
        super().__init__()
        self.torso = TransformerTorso(observation_space, d_model, nhead, num_layers, dim_feedforward, observation_normalizer, history_size, device)
        self.critic_type = critic_type
        self.action_size = action_space.shape[0]
        self.device = device

        if critic_type == "value":
            self.torso_out_layer = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.torso_out_layer = nn.Sequential(nn.Linear(d_model + self.action_size, d_model), nn.GELU())

        if critic_type == "deterministic" or critic_type == "value":
            self.value_layer = nn.Linear(d_model, 1)
        elif critic_type == "distributional":
            self.value_layer = DistributionalValueHead(-150, 150, 51, d_model)
        
        self.to(device)

        # Print parameter count
        print(f"Parameter count critic: {sum(p.numel() for p in self.parameters())}")

    def forward(self, observations, actions=None):
        out = self.torso(observations)
        out = F.gelu(out)
        
        if self.critic_type == "value":
            combined = out
        else:
            combined = torch.cat([out, actions], dim=-1)
            
        out = self.torso_out_layer(combined)
        
        if self.critic_type == "deterministic" or self.critic_type == "value":
            value = self.value_layer(out)
            return torch.squeeze(value, -1)
        else:
            return self.value_layer(out)
