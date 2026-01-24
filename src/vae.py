import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for compressing 128x128 video frames.
    
    Architecture:
    - Encoder: 128x128x3 -> 64x64 -> 32x32 -> 16x16 -> 8x8 (Latent)
    - Latent Dim: 256 (represented as mean/logvar) - SPATIAL (8x8x256)
    - Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128x3
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # ========================
        # Encoder
        # ========================
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        
        # Latent projections (Spatial)
        # Input is 256 channels, Output is latent_dim channels
        self.conv_mu = nn.Conv2d(256, latent_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(256, latent_dim, kernel_size=1)
        
        # ========================
        # Decoder
        # ========================
        # Input: (B, latent_dim, 8, 8)
        
        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            # 16 -> 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            # 32 -> 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # 64 -> 128
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output 0-1
        )

    def encode(self, x):
        """Returns mu, log_var"""
        x = self.encoder(x) # (B, 256, 8, 8)
        mu = self.conv_mu(x) # (B, latent_dim, 8, 8)
        log_var = self.conv_var(x) # (B, latent_dim, 8, 8)
        # Clamp log_var to prevent explosion
        log_var = torch.clamp(log_var, -10, 10)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Sampling trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (B, latent_dim, 8, 8)
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
