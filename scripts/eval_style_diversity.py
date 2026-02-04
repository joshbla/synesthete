import argparse
import os
import sys

import torch

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import load_config, get_device
from vae import VAE
from diffusion import DiffusionTransformer, NoiseScheduler
from audio_gen import AudioGenerator
from audio_features import compute_audio_timeline, audio_feature_dim


def _get_waveform(
    *,
    audio_path: str | None,
    sample_rate: int,
    duration: float,
) -> torch.Tensor:
    if audio_path:
        import soundfile as sf

        wav_data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(wav_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if sr != sample_rate:
            import torchaudio.transforms as T

            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
    else:
        gen = AudioGenerator(sample_rate=sample_rate)
        waveform = gen.generate_sequence(duration=duration, bpm=120)

    target_len = int(sample_rate * duration)
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        waveform = torch.cat([waveform, torch.zeros((1, target_len - waveform.shape[1]))], dim=1)
    return waveform  # (1, N)


def _build_audio_ctx(
    audio_feats: torch.Tensor,  # (T, F)
    *,
    frame_idx: int,
    context: int,
) -> torch.Tensor:
    T = int(audio_feats.shape[0])
    i = int(min(max(frame_idx, 0), T - 1))
    ctx = []
    for j in range(i - context, i + context + 1):
        jj = min(max(j, 0), T - 1)
        ctx.append(audio_feats[jj])
    return torch.stack(ctx, dim=0)  # (T_ctx, F)


def _pairwise_metrics(x: torch.Tensor) -> dict[str, float]:
    """
    x: (K, D) float tensor
    Returns a few simple diversity scalars.
    """
    K = int(x.shape[0])
    if K < 2:
        return {"mean_pairwise_l2": 0.0, "mean_to_mean_l2": 0.0, "mean_nn_l2": 0.0}

    # Mean distance to mean
    mu = x.mean(dim=0, keepdim=True)
    mean_to_mean = torch.sqrt(torch.sum((x - mu) ** 2, dim=-1) + 1e-8).mean().item()

    # Pairwise distances (upper triangle)
    dists = []
    nn = []
    for i in range(K):
        di = torch.sqrt(torch.sum((x[i : i + 1] - x) ** 2, dim=-1) + 1e-8)  # (K,)
        # exclude self for nn
        di_wo = torch.cat([di[:i], di[i + 1 :]], dim=0)
        nn.append(di_wo.min().item())
        for j in range(i + 1, K):
            dists.append(di[j].item())

    mean_pairwise = float(sum(dists) / max(1, len(dists)))
    mean_nn = float(sum(nn) / len(nn))
    return {
        "mean_pairwise_l2": mean_pairwise,
        "mean_to_mean_l2": float(mean_to_mean),
        "mean_nn_l2": mean_nn,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate style diversity for fixed audio.")
    p.add_argument("--override-config", type=str, default=None, help="Override config YAML (e.g., config/smoke.yaml)")
    p.add_argument("--audio", type=str, default=None, help="Optional WAV for conditioning.")
    p.add_argument("--num-styles", type=int, default=16, help="How many styles to sample.")
    p.add_argument("--style-seed-base", type=int, default=0, help="Base seed for style sampling.")
    p.add_argument(
        "--diffusion-seed",
        type=int,
        default=0,
        help="Seed for diffusion sampling noise (kept fixed across styles to isolate style effect).",
    )
    p.add_argument(
        "--frame-idx",
        type=int,
        default=-1,
        help="Which frame index to probe (-1 means middle frame).",
    )
    args = p.parse_args()

    # Ensure outputs/ exists for any downstream tooling (we don't save videos here by default,
    # but keeping outputs centralized avoids root clutter).
    os.makedirs("outputs", exist_ok=True)

    device = get_device()
    config = load_config(override_path=args.override_config) if args.override_config else load_config()

    sample_rate = int(config.get("data", {}).get("sample_rate", 16000))
    fps = int(config.get("data", {}).get("fps", 30))
    duration = float(config.get("data", {}).get("duration", 3.0))

    latent_dim = int(config.get("model", {}).get("latent_dim", 256))
    height = int(config.get("data", {}).get("height", 128))
    latent_spatial_size = height // 16

    num_frames = int(config.get("model", {}).get("num_frames", int(duration * fps)))
    if args.frame_idx == -1:
        frame_idx = max(0, num_frames // 2)
    else:
        frame_idx = int(args.frame_idx)

    num_bands = int(config.get("diffusion", {}).get("audio_feature_num_bands", 8))
    n_fft = int(config.get("diffusion", {}).get("audio_feature_n_fft", 512))
    context = int(config.get("diffusion", {}).get("audio_feature_context", 0))
    include_abs_rms = bool(config.get("diffusion", {}).get("audio_feature_include_abs_rms", False))
    include_onset = bool(config.get("diffusion", {}).get("audio_feature_include_onset", False))
    include_flux_bands = bool(config.get("diffusion", {}).get("audio_feature_include_flux_bands", False))
    timesteps = int(config.get("diffusion", {}).get("timesteps", 50))
    style_dim = int(config.get("diffusion", {}).get("style_dim", 64))
    prev_latent_weight = float(config.get("diffusion", {}).get("prev_latent_weight", 1.0))

    # Load VAE
    vae = VAE(latent_dim=latent_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
    except Exception:
        pass
    vae.eval()

    # Load diffusion model
    d_model = int(config.get("model", {}).get("d_model", 512))
    model = DiffusionTransformer(
        latent_dim=latent_dim,
        d_model=d_model,
        latent_spatial_size=latent_spatial_size,
        audio_feature_dim=audio_feature_dim(
            num_bands,
            include_abs_rms=include_abs_rms,
            include_onset=include_onset,
            include_flux_bands=include_flux_bands,
        ),
        style_dim=style_dim,
        prev_latent_weight=prev_latent_weight,
    ).to(device)
    try:
        model.load_state_dict(torch.load("diffusion_checkpoints/diff_latest.pth", map_location=device))
    except Exception:
        pass
    model.eval()

    scheduler = NoiseScheduler(num_timesteps=timesteps, device=device)

    # Audio conditioning (fixed)
    waveform = _get_waveform(audio_path=args.audio, sample_rate=sample_rate, duration=duration)
    audio_feats = compute_audio_timeline(
        waveform,
        sample_rate=sample_rate,
        fps=fps,
        num_frames=num_frames,
        n_fft=n_fft,
        num_bands=num_bands,
        include_abs_rms=include_abs_rms,
        include_onset=include_onset,
        include_flux_bands=include_flux_bands,
    )  # (T, F)
    audio_ctx = _build_audio_ctx(audio_feats, frame_idx=frame_idx, context=context).to(device)  # (T_ctx, F)
    audio_ctx = audio_ctx.unsqueeze(0)  # (1, T_ctx, F)

    # Sample one latent frame per style (fixed diffusion seed)
    torch.manual_seed(int(args.diffusion_seed))
    prev = torch.zeros((1, latent_dim, latent_spatial_size, latent_spatial_size), device=device)
    frame_idx_t = torch.tensor([int(frame_idx)], device=device, dtype=torch.long)
    shape = (1, latent_dim, latent_spatial_size, latent_spatial_size)

    latents = []
    frames = []
    for k in range(int(args.num_styles)):
        # Style varies, everything else fixed
        torch.manual_seed(int(args.style_seed_base) + k)
        style = torch.randn((1, style_dim), device=device)

        # Ensure diffusion noise is identical across styles by reseeding before sampling.
        torch.manual_seed(int(args.diffusion_seed))
        lat = scheduler.sample(model, audio_ctx, shape, style=style, prev_latent=prev, frame_idx=frame_idx_t)
        latents.append(lat.squeeze(0))
        with torch.no_grad():
            fr = vae.decode(lat).clamp(0, 1)  # (1, 3, H, W)
        frames.append(fr.squeeze(0))

    latents_t = torch.stack(latents, dim=0)  # (K, C, h, w)
    frames_t = torch.stack(frames, dim=0)    # (K, 3, H, W)

    # Diversity metrics
    lat_vec = latents_t.flatten(1)  # (K, D)
    fr_vec = frames_t.flatten(1)    # (K, D')
    lat_m = _pairwise_metrics(lat_vec)
    fr_m = _pairwise_metrics(fr_vec)

    print("Style diversity probe (fixed audio, varying style):")
    print(f"- num_styles: {int(args.num_styles)}")
    print(f"- probed_frame_idx: {int(frame_idx)} / {int(num_frames)}")
    print("Latent diversity (L2):")
    print(f"- mean_pairwise_l2: {lat_m['mean_pairwise_l2']:.6f}")
    print(f"- mean_to_mean_l2:  {lat_m['mean_to_mean_l2']:.6f}")
    print(f"- mean_nn_l2:       {lat_m['mean_nn_l2']:.6f}")
    print("Decoded-frame diversity (L2):")
    print(f"- mean_pairwise_l2: {fr_m['mean_pairwise_l2']:.6f}")
    print(f"- mean_to_mean_l2:  {fr_m['mean_to_mean_l2']:.6f}")
    print(f"- mean_nn_l2:       {fr_m['mean_nn_l2']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

