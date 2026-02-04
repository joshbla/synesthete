import argparse
from pathlib import Path
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


def _build_audio_batch(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    fps: int,
    num_frames: int,
    n_fft: int,
    num_bands: int,
    context: int,
    include_abs_rms: bool,
    include_onset: bool,
    include_flux_bands: bool,
    device: torch.device,
) -> torch.Tensor:
    audio_feats = compute_audio_timeline(
        waveform.cpu(),  # (1, N)
        sample_rate=sample_rate,
        fps=fps,
        num_frames=num_frames,
        n_fft=n_fft,
        num_bands=num_bands,
        normalize_per_clip=bool(os.environ.get("SYNESTHETE_AUDIO_NORM", "1") == "1"),
        include_abs_rms=include_abs_rms,
        include_onset=include_onset,
        include_flux_bands=include_flux_bands,
    )  # (T, F)

    ctx_feats = []
    for i in range(num_frames):
        ctx = []
        for j in range(i - context, i + context + 1):
            jj = min(max(j, 0), num_frames - 1)
            ctx.append(audio_feats[jj])
        ctx_feats.append(torch.stack(ctx, dim=0))
    return torch.stack(ctx_feats, dim=0).to(device)  # (T, T_ctx, F)


def _sample_latents_sequential(
    *,
    model: DiffusionTransformer,
    scheduler: NoiseScheduler,
    audio_batch: torch.Tensor,  # (T, T_ctx, F)
    style: torch.Tensor,  # (T, style_dim)
    latent_dim: int,
    latent_spatial_size: int,
    diffusion_seed: int,
    audio_guidance_scale: float,
) -> torch.Tensor:
    device = audio_batch.device
    T = int(audio_batch.shape[0])

    # Make the whole run deterministic (the scheduler uses torch.randn internally).
    torch.manual_seed(int(diffusion_seed))

    latents_out = []
    prev = torch.zeros((1, latent_dim, latent_spatial_size, latent_spatial_size), device=device)
    for i in range(T):
        frame_audio = audio_batch[i : i + 1]
        frame_style = style[i : i + 1]
        frame_idx = torch.tensor([i], device=device, dtype=torch.long)
        shape = (1, latent_dim, latent_spatial_size, latent_spatial_size)
        lat = scheduler.sample(
            model,
            frame_audio,
            shape,
            style=frame_style,
            prev_latent=prev,
            frame_idx=frame_idx,
            audio_guidance_scale=audio_guidance_scale,
        )
        latents_out.append(lat)
        prev = lat.detach()
    return torch.cat(latents_out, dim=0)  # (T, C, H, W)


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate audio reactivity via an audio-shuffle test.")
    p.add_argument("--override-config", type=str, default=None, help="Path to override config YAML (e.g., config/smoke.yaml)")
    p.add_argument("--audio", type=str, default=None, help="Optional path to input WAV for conditioning.")
    p.add_argument("--num-frames", type=int, default=None, help="Override number of frames (defaults to config.model.num_frames)")
    p.add_argument("--diffusion-seed", type=int, default=0, help="Seed for diffusion sampling noise (deterministic A/B).")
    p.add_argument("--shuffle-seed", type=int, default=0, help="Seed for audio shuffling permutation.")
    p.add_argument("--save-videos", action="store_true", help="Also write mp4s for visual inspection.")
    p.add_argument(
        "--out-prefix",
        type=str,
        default=os.path.join("outputs", "output_reactivity"),
        help="Prefix for outputs when --save-videos is set.",
    )
    args = p.parse_args()

    device = get_device()
    config = load_config(override_path=args.override_config) if args.override_config else load_config()

    sample_rate = int(config.get("data", {}).get("sample_rate", 16000))
    fps = int(config.get("data", {}).get("fps", 30))
    duration = float(config.get("data", {}).get("duration", 3.0))

    latent_dim = int(config.get("model", {}).get("latent_dim", 256))
    height = int(config.get("data", {}).get("height", 128))
    latent_spatial_size = height // 16

    num_bands = int(config.get("diffusion", {}).get("audio_feature_num_bands", 8))
    n_fft = int(config.get("diffusion", {}).get("audio_feature_n_fft", 512))
    context = int(config.get("diffusion", {}).get("audio_feature_context", 0))
    include_abs_rms = bool(config.get("diffusion", {}).get("audio_feature_include_abs_rms", False))
    include_onset = bool(config.get("diffusion", {}).get("audio_feature_include_onset", False))
    include_flux_bands = bool(config.get("diffusion", {}).get("audio_feature_include_flux_bands", False))
    timesteps = int(config.get("diffusion", {}).get("timesteps", 50))
    style_dim = int(config.get("diffusion", {}).get("style_dim", 64))

    if args.num_frames is None:
        num_frames = int(config.get("model", {}).get("num_frames", int(duration * fps)))
    else:
        num_frames = int(args.num_frames)

    # Load VAE (for optional decoding)
    vae = VAE(latent_dim=latent_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
    except Exception:
        pass
    vae.eval()

    # Load diffusion model
    d_model = int(config.get("model", {}).get("d_model", 512))
    prev_latent_weight = float(config.get("diffusion", {}).get("prev_latent_weight", 1.0))
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

    # Audio
    if args.audio:
        import soundfile as sf

        wav_data, sr = sf.read(args.audio)
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

    # Ensure exact length (1, N)
    target_len = int(sample_rate * duration)
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        waveform = torch.cat([waveform, torch.zeros((1, target_len - waveform.shape[1]))], dim=1)

    # Build conditioning batch (T, T_ctx, F)
    # Respect config for per-clip normalization
    os.environ["SYNESTHETE_AUDIO_NORM"] = "1" if bool(config.get("diffusion", {}).get("audio_feature_normalize_per_clip", True)) else "0"

    audio_batch = _build_audio_batch(
        waveform,
        sample_rate=sample_rate,
        fps=fps,
        num_frames=num_frames,
        n_fft=n_fft,
        num_bands=num_bands,
        context=context,
        include_abs_rms=include_abs_rms,
        include_onset=include_onset,
        include_flux_bands=include_flux_bands,
        device=device,
    )

    # Style: fixed per run, repeated per frame (to isolate audio)
    style_mode = (config.get("inference", {}) or {}).get("style_mode", "random")
    style_seed = (config.get("inference", {}) or {}).get("style_seed", None)
    if style_seed is not None:
        torch.manual_seed(int(style_seed))
    if style_mode == "zero":
        style_one = torch.zeros((1, style_dim), device=device)
    else:
        style_one = torch.randn((1, style_dim), device=device)
    style = style_one.repeat(int(audio_batch.shape[0]), 1)

    # Normal sampling
    audio_guidance_scale = float((config.get("inference", {}) or {}).get("audio_guidance_scale", 1.0))
    lat_norm = _sample_latents_sequential(
        model=model,
        scheduler=scheduler,
        audio_batch=audio_batch,
        style=style,
        latent_dim=latent_dim,
        latent_spatial_size=latent_spatial_size,
        diffusion_seed=args.diffusion_seed,
        audio_guidance_scale=audio_guidance_scale,
    )

    # Shuffled-audio sampling (same diffusion/style seeds)
    g = torch.Generator(device="cpu").manual_seed(int(args.shuffle_seed))
    perm = torch.randperm(int(audio_batch.shape[0]), generator=g)
    audio_batch_shuf = audio_batch[perm]
    lat_shuf = _sample_latents_sequential(
        model=model,
        scheduler=scheduler,
        audio_batch=audio_batch_shuf,
        style=style,
        latent_dim=latent_dim,
        latent_spatial_size=latent_spatial_size,
        diffusion_seed=args.diffusion_seed,
        audio_guidance_scale=audio_guidance_scale,
    )

    # Metrics (latent space): how much the output changes when audio is shuffled
    l2 = torch.mean((lat_norm - lat_shuf) ** 2).item()
    l1 = torch.mean(torch.abs(lat_norm - lat_shuf)).item()
    denom = (torch.mean(torch.abs(lat_norm)) + 1e-8).item()
    rel_l1 = float(l1 / denom)

    print("Audio reactivity (shuffle test):")
    print(f"- latent_mse: {l2:.6f}")
    print(f"- latent_l1:  {l1:.6f}")
    print(f"- rel_l1:     {rel_l1:.6f}   (higher = more change when audio is shuffled)")

    # Metrics (decoded frames): more human-meaningful than latent deltas.
    # If VAE isn't available, skip silently (latent metrics still useful).
    try:
        with torch.no_grad():
            frames_a = vae.decode(lat_norm).clamp(0, 1)
            frames_b = vae.decode(lat_shuf).clamp(0, 1)
        f_mse = torch.mean((frames_a - frames_b) ** 2).item()
        f_l1 = torch.mean(torch.abs(frames_a - frames_b)).item()
        f_denom = (torch.mean(torch.abs(frames_a)) + 1e-8).item()
        f_rel_l1 = float(f_l1 / f_denom)
        print("Decoded-frame reactivity (shuffle test):")
        print(f"- frame_mse: {f_mse:.6f}")
        print(f"- frame_l1:  {f_l1:.6f}")
        print(f"- rel_l1:    {f_rel_l1:.6f}   (higher = more visible change)")
    except Exception:
        pass

    if args.save_videos:
        from torchcodec.encoders import VideoEncoder
        import torchaudio
        import subprocess

        out_prefix = args.out_prefix
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        out_a = f"{out_prefix}_normal.mp4"
        out_b = f"{out_prefix}_shuffled.mp4"

        with torch.no_grad():
            frames_a = vae.decode(lat_norm).clamp(0, 1).cpu()
            frames_b = vae.decode(lat_shuf).clamp(0, 1).cpu()

        def _write(frames: torch.Tensor, path: str) -> None:
            video_uint8 = (frames * 255).byte()
            enc = VideoEncoder(video_uint8, frame_rate=fps)
            tmp_video = path.replace(".mp4", "_tmp.mp4")
            tmp_audio = path.replace(".mp4", "_tmp.wav")
            enc.to_file(tmp_video)
            torchaudio.save(tmp_audio, waveform, sample_rate)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_video, "-i", tmp_audio, "-c:v", "copy", "-c:a", "aac", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            Path(tmp_video).unlink(missing_ok=True)
            Path(tmp_audio).unlink(missing_ok=True)

        _write(frames_a, out_a)
        _write(frames_b, out_b)
        print(f"Wrote: {out_a}")
        print(f"Wrote: {out_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

