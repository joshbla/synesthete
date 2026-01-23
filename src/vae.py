import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for compressing 128x128 video frames.
    
    Architecture:
    - Encoder: 128x128x3 -> 64x64 -> 32x32 -> 16x16 -> 8x8 (Latent)
    - Latent Dim: 256 (represented as mean/logvar)
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
        
        # Flatten size: 256 channels * 8 * 8 = 16384
        self.flatten_size = 256 * 8 * 8
        
        # Latent projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        
        # ========================
        # Decoder
        # ========================
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
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
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        # Clamp log_var to prevent explosion
        log_var = torch.clamp(log_var, -10, 10)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Sampling trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
