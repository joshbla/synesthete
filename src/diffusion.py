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
        
    def sample(self, model, audio, shape):
        """
        Sample from the model.
        audio: (B, 1, 48000)
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
                predicted_noise = model(x, t, audio)
                
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
    - audio: Raw Audio (B, 1, 48000)
    
    Output:
    - noise_pred: Predicted Noise (B, 256, 8, 8)
    """
    def __init__(self, latent_dim=256, d_model=512, nhead=8, num_layers=6, latent_spatial_size=8):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.latent_spatial_size = latent_spatial_size
        num_tokens = latent_spatial_size * latent_spatial_size
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 2. Audio Encoder (Same as AudioToVideoNet)
        # Input: (B, 1, 48000) -> Output: (B, d_model, T_compressed)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38),
            nn.GELU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, d_model, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        
        # 3. Latent Input Projection
        # Flatten spatial grid to tokens
        self.input_proj = nn.Conv2d(latent_dim, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        
        # 4. Transformer Decoder (Cross-Attn to Audio)
        # We use TransformerDecoder because we have "queries" (latents) and "keys/values" (audio)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 5. Output Projection
        self.output_proj = nn.Conv2d(d_model, latent_dim, kernel_size=1)
        
    def forward(self, x, t, audio):
        """
        x: (B, latent_dim, H, W)
        t: (B,)
        audio: (B, 1, 48000)
        """
        B = x.shape[0]
        
        # 1. Embed Time
        t_emb = self.time_mlp(t) # (B, d_model)
        t_emb = t_emb.unsqueeze(1) # (B, 1, d_model)
        
        # 2. Embed Audio
        audio_feats = self.audio_encoder(audio) # (B, d_model, T_seq)
        audio_feats = audio_feats.permute(0, 2, 1) # (B, T_seq, d_model)
        
        # 3. Embed Latents
        h = self.input_proj(x) # (B, d_model, H, W)
        h = h.flatten(2).permute(0, 2, 1) # (B, num_tokens, d_model)
        
        # Add positional embedding
        h = h + self.pos_embed
        
        # Add time embedding (broadcast over sequence)
        h = h + t_emb
        
        # 4. Transformer
        # tgt = latents, memory = audio
        h = self.transformer(tgt=h, memory=audio_feats) # (B, num_tokens, d_model)
        
        # 5. Output
        h = h.permute(0, 2, 1).view(B, self.d_model, self.latent_spatial_size, self.latent_spatial_size)
        out = self.output_proj(h) # (B, latent_dim, H, W)
        
        return out
