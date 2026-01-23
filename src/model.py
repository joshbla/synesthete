import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer temporal awareness."""
    
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AudioToVideoNet(nn.Module):
    """
    Hybrid CNN + Transformer architecture for audio-to-video synthesis.
    
    Architecture:
        1. CNN Audio Encoder: Efficiently extracts local audio features
        2. Transformer Temporal Module: Models long-range dependencies across time
        3. CNN Frame Decoder: Generates video frames from temporally-aware features
    
    This hybrid approach combines:
        - CNN efficiency for local pattern extraction (beats, attacks, harmonics)
        - Transformer expressivity for long-range temporal relationships
          (connecting musical phrases across the full 3-second window)
    """
    
    def __init__(self, num_frames=90, frame_size=(128, 128), 
                 d_model=256, nhead=8, num_transformer_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.h, self.w = frame_size
        self.d_model = d_model
        
        # ============================================
        # Audio Encoder (CNN): Local feature extraction
        # ============================================
        # Input: (B, 1, T_audio) where T_audio = 48000 (3s @ 16kHz)
        # Output: (B, d_model, T_compressed)
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
        
        # ============================================
        # Temporal Transformer: Long-range dependencies
        # ============================================
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_frames, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Input/output: (B, T, D)
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Layer norm after transformer
        self.post_transformer_norm = nn.LayerNorm(d_model)
        
        # ============================================
        # Frame Decoder (CNN-based): Spatial generation
        # ============================================
        # Instead of pure MLP, use a small CNN decoder for better spatial coherence
        # Input: (B * T, d_model) -> reshape to (B * T, d_model, 1, 1) -> upsample
        
        # First project to spatial seed
        self.spatial_seed_size = 8  # Start from 8x8
        self.to_spatial = nn.Linear(d_model, d_model * self.spatial_seed_size * self.spatial_seed_size)
        
        # CNN Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.frame_decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output 0-1 for pixels
        )

    def forward(self, audio):
        """
        Forward pass.
        
        Args:
            audio: (B, 1, T_audio) raw audio waveform
            
        Returns:
            video: (B, num_frames, 3, H, W) generated video frames
        """
        B = audio.size(0)
        
        # 1. CNN Audio Encoding
        x = self.audio_encoder(audio)  # (B, d_model, T_compressed)
        
        # 2. Align to target frame count
        x = F.interpolate(x, size=self.num_frames, mode='linear', align_corners=False)
        
        # 3. Prepare for Transformer (B, T, D)
        x = x.permute(0, 2, 1)  # (B, num_frames, d_model)
        
        # Save local features for residual connection
        x_local = x
        
        # 4. Add positional encoding
        x = self.positional_encoding(x)
        
        # 5. Transformer temporal modeling
        x = self.temporal_transformer(x)  # (B, num_frames, d_model)
        x = self.post_transformer_norm(x)
        
        # Residual connection: Mix local features with temporal features
        # This ensures the model doesn't lose the "beat" (local info) while adding "flow" (temporal info)
        x = x + x_local
        
        # 6. Decode to frames
        # Reshape for frame-wise decoding
        x = x.reshape(B * self.num_frames, self.d_model)  # (B*T, d_model)
        
        # Project to spatial seed
        x = self.to_spatial(x)  # (B*T, d_model * 8 * 8)
        x = x.view(B * self.num_frames, self.d_model, self.spatial_seed_size, self.spatial_seed_size)
        
        # CNN decode to full resolution
        x = self.frame_decoder(x)  # (B*T, 3, H, W)
        
        # 7. Reshape to video
        x = x.view(B, self.num_frames, 3, self.h, self.w)
        
        return x


# Keep the old model for reference/comparison
class AudioToVideoNetLegacy(nn.Module):
    """Original CNN + MLP architecture (kept for comparison)."""
    
    def __init__(self, num_frames=90, frame_size=(128, 128)):
        super().__init__()
        self.num_frames = num_frames
        self.h, self.w = frame_size
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * self.h * self.w),
            nn.Sigmoid()
        )

    def forward(self, audio):
        x = self.encoder(audio)
        x = F.interpolate(x, size=self.num_frames, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = x.view(x.size(0), self.num_frames, 3, self.h, self.w)
        return x
