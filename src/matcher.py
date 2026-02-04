import torch
import torch.nn as nn


class AudioLatentMatcher(nn.Module):
    """
    Lightweight matcher / discriminator for (audio_features, latent) pairs.

    Intended use (Phase 5):
    - Train it to classify matched vs mismatched pairs.
    - Use its score as an auxiliary signal to encourage diffusion outputs to depend on audio.

    Inputs:
    - audio: (B, T_ctx, F)
    - latent: (B, C, H, W)  (typically VAE latent space)

    Output:
    - logits: (B,) (higher = "matched")
    """

    def __init__(
        self,
        *,
        audio_feature_dim: int,
        latent_dim: int,
        d_model: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.audio_feature_dim = int(audio_feature_dim)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(self.audio_feature_dim),
            nn.Linear(self.audio_feature_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.d_model, self.d_model),
        )

        self.latent_proj = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.d_model, kernel_size=1),
            nn.GELU(),
        )

        # Combine pooled representations and classify
        combo_dim = self.d_model * 4  # [a, z, |a-z|, a*z]
        self.head = nn.Sequential(
            nn.LayerNorm(combo_dim),
            nn.Linear(combo_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, audio: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"Expected audio shape (B, T_ctx, F), got {tuple(audio.shape)}")
        if latent.ndim != 4:
            raise ValueError(f"Expected latent shape (B, C, H, W), got {tuple(latent.shape)}")
        if audio.shape[-1] != self.audio_feature_dim:
            raise ValueError(
                f"Expected audio feature dim {self.audio_feature_dim}, got {audio.shape[-1]}"
            )
        if latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {latent.shape[1]}")

        # Audio: project per-time then mean-pool over time
        a = self.audio_proj(audio)  # (B, T, D)
        a = a.mean(dim=1)  # (B, D)

        # Latent: project then mean-pool over spatial grid
        z = self.latent_proj(latent)  # (B, D, H, W)
        z = z.mean(dim=(2, 3))  # (B, D)

        combo = torch.cat([a, z, (a - z).abs(), a * z], dim=-1)
        logits = self.head(combo).squeeze(-1)  # (B,)
        return logits

