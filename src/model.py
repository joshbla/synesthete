import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioToVideoNet(nn.Module):
    def __init__(self, num_frames=90, frame_size=(128, 128)):
        super().__init__()
        self.num_frames = num_frames
        self.h, self.w = frame_size
        
        # Audio Encoder: Compresses raw audio into temporal features
        # Input: (B, 1, T_audio)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38), # Downsample
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
        
        # Frame Decoder: Projects features to pixels
        # Project 256 channels -> 3 * H * W pixels
        # We'll use a slightly larger linear layer
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * self.h * self.w),
            nn.Sigmoid() # Output 0-1 for pixels
        )

    def forward(self, audio):
        # audio: (B, 1, T)
        
        # 1. Encode
        x = self.encoder(audio) # (B, 256, T_compressed)
        
        # 2. Align time dimension
        # Use interpolate instead of AdaptiveAvgPool1d for MPS compatibility
        x = F.interpolate(x, size=self.num_frames, mode='linear', align_corners=False)
        
        # 3. Prepare for decoding
        x = x.permute(0, 2, 1) # (B, num_frames, 256)
        
        # 4. Decode to frames
        x = self.decoder(x) # (B, num_frames, 3*H*W)
        
        # 5. Reshape
        x = x.view(x.size(0), self.num_frames, 3, self.h, self.w)
        
        return x
