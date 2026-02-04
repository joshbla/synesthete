import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class NoiseScheduler:
    def __init__(self, num_timesteps=50, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_0, t):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        return x_t, noise
        
    def sample(self, model, audio, shape, style=None, prev_latent=None, frame_idx=None):
        """
        Sample from the model.
        audio: (B, T_audio, F) frame-aligned audio features
        style: (B, style_dim) optional clip-level style latent
        prev_latent: (B, C, H, W) optional previous-frame latent (Phase 4)
        frame_idx: (B,) optional frame index (Phase 4)
        shape: (B, C, H, W) -> (B, 256, 8, 8)
        """
        model.eval()
        B = shape[0]
        
        # Start with pure noise
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.num_timesteps)):
            if i % 100 == 0:
                print(f"Sampling timestep {i}/{self.num_timesteps}...", flush=True)
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = model(x, t, audio, style=style, prev_latent=prev_latent, frame_idx=frame_idx)
                
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # DDPM Sampling formula
            # x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_cumprod) * eps_theta) + sigma * z
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise)
            x = x + torch.sqrt(beta) * noise
            
        return x

class DiffusionTransformer(nn.Module):
    """
    Transformer-based Diffusion Model.
    
    Inputs:
    - x: Noisy Latents (B, 256, 8, 8)
    - t: Timesteps (B,)
    - audio: Frame-aligned audio features (B, T_audio, F)
    
    Output:
    - noise_pred: Predicted Noise (B, 256, 8, 8)
    """
    def __init__(
        self,
        latent_dim=256,
        d_model=512,
        nhead=8,
        num_layers=6,
        latent_spatial_size=8,
        audio_feature_dim=13,
        style_dim=64,
        prev_latent_weight: float = 1.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.latent_spatial_size = latent_spatial_size
        self.audio_feature_dim = audio_feature_dim
        self.style_dim = style_dim
        self.prev_latent_weight = float(prev_latent_weight)
        num_tokens = latent_spatial_size * latent_spatial_size
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 2. Audio conditioning projection
        # Input: (B, T_audio, F) -> Output: (B, T_audio, d_model)
        # Phase 1 goal is stable, time-aligned conditioning (not raw waveform encoding).
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_feature_dim),
            nn.Linear(audio_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding so the model can use ordering in T_audio
        self.audio_positional_encoding = PositionalEncoding(d_model=d_model, max_len=512, dropout=0.0)

        # Style token: a single clip-level latent that provides a place to put
        # aesthetic randomness (Phase 2 factorization).
        self.style_mlp = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Phase 4: frame index embedding (separate from diffusion timestep)
        self.frame_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3. Latent Input Projection
        # Flatten spatial grid to tokens
        self.input_proj = nn.Conv2d(latent_dim, d_model, kernel_size=1)
        # Phase 4: previous latent projection (same spatial shape as x)
        self.prev_proj = nn.Conv2d(latent_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        
        # 4. Transformer Decoder (Cross-Attn to Audio)
        # We use TransformerDecoder because we have "queries" (latents) and "keys/values" (audio)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 5. Output Projection
        self.output_proj = nn.Conv2d(d_model, latent_dim, kernel_size=1)
        
    def forward(self, x, t, audio, style=None, prev_latent=None, frame_idx=None):
        """
        x: (B, latent_dim, H, W)
        t: (B,)
        audio: (B, T_audio, F)
        style: (B, style_dim) or None
        prev_latent: (B, latent_dim, H, W) or None
        frame_idx: (B,) int/float or None
        """
        B = x.shape[0]
        
        # 1. Embed Time
        t_emb = self.time_mlp(t) # (B, d_model)
        t_emb = t_emb.unsqueeze(1) # (B, 1, d_model)

        # 1b. Embed Frame Index
        if frame_idx is None:
            frame_idx = torch.zeros((B,), device=x.device, dtype=torch.long)
        if frame_idx.ndim != 1 or frame_idx.shape[0] != B:
            raise ValueError(f"Expected frame_idx shape (B,), got {tuple(frame_idx.shape)}")
        f_emb = self.frame_mlp(frame_idx.to(torch.float32))  # (B, d_model)
        f_emb = f_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # 2. Embed Audio features
        if audio.ndim != 3:
            raise ValueError(f"Expected audio features shape (B, T_audio, F), got {tuple(audio.shape)}")
        if audio.shape[-1] != self.audio_feature_dim:
            raise ValueError(
                f"Expected audio feature dim {self.audio_feature_dim}, got {audio.shape[-1]}. "
                f"Check diffusion.audio_feature_num_bands / model init."
            )
        audio_feats = self.audio_proj(audio)  # (B, T_audio, d_model)
        audio_feats = self.audio_positional_encoding(audio_feats)

        # 2b. Style token
        if style is None:
            style = torch.zeros((B, self.style_dim), device=audio.device, dtype=audio.dtype)
        if style.ndim != 2 or style.shape[0] != B or style.shape[1] != self.style_dim:
            raise ValueError(
                f"Expected style shape (B, {self.style_dim}) where B={B}, got {tuple(style.shape)}"
            )
        style_tok = self.style_mlp(style).unsqueeze(1)  # (B, 1, d_model)

        # Cross-attend to [style_tok, audio_feats]
        memory = torch.cat([style_tok, audio_feats], dim=1)  # (B, 1+T_audio, d_model)
        
        # 3. Embed Latents
        h = self.input_proj(x) # (B, d_model, H, W)
        h = h.flatten(2).permute(0, 2, 1) # (B, num_tokens, d_model)

        # 3b. Previous latent conditioning (additive)
        if prev_latent is None:
            prev_latent = torch.zeros_like(x)
        if prev_latent.shape != x.shape:
            raise ValueError(f"Expected prev_latent shape {tuple(x.shape)}, got {tuple(prev_latent.shape)}")
        prev_h = self.prev_proj(prev_latent).flatten(2).permute(0, 2, 1)  # (B, num_tokens, d_model)
        
        # Add positional embedding
        h = h + self.pos_embed
        
        # Add time + frame embeddings (broadcast over sequence) and prev conditioning
        h = h + t_emb + f_emb + (self.prev_latent_weight * prev_h)
        
        # 4. Transformer
        # tgt = latents, memory = (style + audio)
        h = self.transformer(tgt=h, memory=memory) # (B, num_tokens, d_model)
        
        # 5. Output
        h = h.permute(0, 2, 1).view(B, self.d_model, self.latent_spatial_size, self.latent_spatial_size)
        out = self.output_proj(h) # (B, latent_dim, H, W)
        
        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence ordering."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
