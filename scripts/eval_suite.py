import argparse
import os
import sys
from dataclasses import dataclass

import torch

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import load_config, get_device
from vae import VAE
from diffusion import DiffusionTransformer, NoiseScheduler
from audio_gen import AudioGenerator
from audio_features import compute_audio_timeline, audio_feature_dim


@dataclass(frozen=True)
class ReactivityMetrics:
    frame_rel_l1: float
    frame_l1: float
    latent_rel_l1: float
    latent_l1: float


def _mk_audio_variants(sample_rate: int, duration: float) -> dict[str, torch.Tensor]:
    gen = AudioGenerator(sample_rate=sample_rate)
    base = gen.generate_sequence(duration=duration, bpm=120)  # amp+rhythm-ish
    dropouts = gen.generate_sequence_dropouts(duration=duration, bpm=120, full_silence=True)

    # Band-rich: alternate low/high tones
    t = torch.linspace(0, duration, int(sample_rate * duration))
    f1, f2 = 110.0, 1760.0
    gate = (torch.sin(2 * torch.pi * 2.0 * t) > 0).to(torch.float32)  # 2Hz alternation
    tone = torch.sin(2 * torch.pi * (f1 * t)) * gate + torch.sin(2 * torch.pi * (f2 * t)) * (1.0 - gate)
    band_rich = tone.unsqueeze(0) * 0.5

    # Onset-heavy: sparse clicks
    clicks = torch.zeros_like(t)
    step = max(1, int(sample_rate * 0.18))
    clicks[::step] = 1.0
    onset = torch.nn.functional.conv1d(
        clicks.view(1, 1, -1),
        torch.hann_window(63).view(1, 1, -1),
        padding=31,
    ).view(-1)
    onset = onset / (onset.abs().max() + 1e-8)
    onset = onset.unsqueeze(0) * 0.6

    # Semantic-ish: jaws theme
    jaws = gen.generate_jaws_theme(duration=duration)

    return {
        "sequence": base,
        "silence_dropouts": dropouts,
        "band_rich": band_rich,
        "onset": onset,
        "jaws": jaws,
    }


def _build_audio_batch(
    waveform: torch.Tensor,  # (1, N)
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
    feats = compute_audio_timeline(
        waveform,
        sample_rate=sample_rate,
        fps=fps,
        num_frames=num_frames,
        n_fft=n_fft,
        num_bands=num_bands,
        normalize_per_clip=bool(cfg.get("diffusion", {}).get("audio_feature_normalize_per_clip", True)),
        include_abs_rms=include_abs_rms,
        include_onset=include_onset,
        include_flux_bands=include_flux_bands,
    )  # (T, F)
    ctx_feats = []
    for i in range(num_frames):
        ctx = []
        for j in range(i - context, i + context + 1):
            jj = min(max(j, 0), num_frames - 1)
            ctx.append(feats[jj])
        ctx_feats.append(torch.stack(ctx, dim=0))
    return torch.stack(ctx_feats, dim=0).to(device)


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
    torch.manual_seed(int(diffusion_seed))
    out = []
    prev = torch.zeros((1, latent_dim, latent_spatial_size, latent_spatial_size), device=device)
    for i in range(T):
        frame_audio = audio_batch[i : i + 1]
        frame_style = style[i : i + 1]
        frame_idx = torch.tensor([i], device=device, dtype=torch.long)
        lat = scheduler.sample(
            model,
            frame_audio,
            (1, latent_dim, latent_spatial_size, latent_spatial_size),
            style=frame_style,
            prev_latent=prev,
            frame_idx=frame_idx,
            audio_guidance_scale=audio_guidance_scale,
        )
        out.append(lat)
        prev = lat.detach()
    return torch.cat(out, dim=0)


def _shuffle_T(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(int(x.shape[0]), generator=g)
    return x[perm]


def main() -> int:
    p = argparse.ArgumentParser(description="Run a small evaluation suite for audio reactivity + diversity.")
    p.add_argument("--override-config", type=str, default=None)
    p.add_argument("--diffusion-seed", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--save-best-pair", action="store_true", help="Save 1 A/B mp4 pair for the most reactive variant.")
    p.add_argument("--out-prefix", type=str, default=os.path.join("outputs", "suite_reactivity"))
    args = p.parse_args()

    device = get_device()
    cfg = load_config(override_path=args.override_config) if args.override_config else load_config()
    os.makedirs("outputs", exist_ok=True)

    sample_rate = int(cfg.get("data", {}).get("sample_rate", 16000))
    fps = int(cfg.get("data", {}).get("fps", 30))
    duration = float(cfg.get("data", {}).get("duration", 3.0))
    num_frames = int(cfg.get("model", {}).get("num_frames", int(duration * fps)))
    height = int(cfg.get("data", {}).get("height", 128))
    latent_spatial_size = height // 16
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 256))
    d_model = int(cfg.get("model", {}).get("d_model", 512))

    num_bands = int(cfg.get("diffusion", {}).get("audio_feature_num_bands", 8))
    n_fft = int(cfg.get("diffusion", {}).get("audio_feature_n_fft", 512))
    context = int(cfg.get("diffusion", {}).get("audio_feature_context", 0))
    include_abs_rms = bool(cfg.get("diffusion", {}).get("audio_feature_include_abs_rms", False))
    include_onset = bool(cfg.get("diffusion", {}).get("audio_feature_include_onset", False))
    include_flux_bands = bool(cfg.get("diffusion", {}).get("audio_feature_include_flux_bands", False))
    timesteps = int(cfg.get("diffusion", {}).get("timesteps", 50))
    style_dim = int(cfg.get("diffusion", {}).get("style_dim", 64))
    prev_latent_weight = float(cfg.get("diffusion", {}).get("prev_latent_weight", 1.0))
    audio_guidance_scale = float((cfg.get("inference", {}) or {}).get("audio_guidance_scale", 1.0))

    vae = VAE(latent_dim=latent_dim).to(device)
    try:
        vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
    except Exception:
        pass
    vae.eval()

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

    # Style: fixed (zero) for reactivity tests
    style = torch.zeros((num_frames, style_dim), device=device)

    variants = _mk_audio_variants(sample_rate, duration)
    results: dict[str, ReactivityMetrics] = {}

    best_name = None
    best_score = -1.0
    best_pair = None

    for name, wav in variants.items():
        audio_batch = _build_audio_batch(
            wav,
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
        lat_a = _sample_latents_sequential(
            model=model,
            scheduler=scheduler,
            audio_batch=audio_batch,
            style=style,
            latent_dim=latent_dim,
            latent_spatial_size=latent_spatial_size,
            diffusion_seed=args.diffusion_seed,
            audio_guidance_scale=audio_guidance_scale,
        )
        lat_b = _sample_latents_sequential(
            model=model,
            scheduler=scheduler,
            audio_batch=_shuffle_T(audio_batch, seed=args.shuffle_seed),
            style=style,
            latent_dim=latent_dim,
            latent_spatial_size=latent_spatial_size,
            diffusion_seed=args.diffusion_seed,
            audio_guidance_scale=audio_guidance_scale,
        )

        latent_l1 = torch.mean(torch.abs(lat_a - lat_b)).item()
        latent_rel = float(latent_l1 / (torch.mean(torch.abs(lat_a)) + 1e-8).item())

        with torch.no_grad():
            fr_a = vae.decode(lat_a).clamp(0, 1)
            fr_b = vae.decode(lat_b).clamp(0, 1)
        frame_l1 = torch.mean(torch.abs(fr_a - fr_b)).item()
        frame_rel = float(frame_l1 / (torch.mean(torch.abs(fr_a)) + 1e-8).item())

        results[name] = ReactivityMetrics(
            frame_rel_l1=frame_rel,
            frame_l1=frame_l1,
            latent_rel_l1=latent_rel,
            latent_l1=latent_l1,
        )

        if frame_rel > best_score:
            best_score = frame_rel
            best_name = name
            best_pair = (fr_a.detach().cpu(), fr_b.detach().cpu(), wav.detach().cpu())

    print("Eval suite: audio shuffle reactivity (decoded-frame rel_l1; higher is better):")
    for k in sorted(results.keys()):
        m = results[k]
        print(f"- {k}: frame_rel_l1={m.frame_rel_l1:.6f} frame_l1={m.frame_l1:.6f} latent_rel_l1={m.latent_rel_l1:.6f}")

    if args.save_best_pair and best_pair is not None and best_name is not None:
        from torchcodec.encoders import VideoEncoder
        import torchaudio
        import subprocess
        from pathlib import Path

        out_prefix = f"{args.out_prefix}_{best_name}"
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        out_a = f"{out_prefix}_normal.mp4"
        out_b = f"{out_prefix}_shuffled.mp4"
        fr_a, fr_b, wav = best_pair

        def _write(frames: torch.Tensor, path: str) -> None:
            video_uint8 = (frames * 255).byte()
            enc = VideoEncoder(video_uint8, frame_rate=fps)
            tmp_video = path.replace(".mp4", "_tmp.mp4")
            tmp_audio = path.replace(".mp4", "_tmp.wav")
            enc.to_file(tmp_video)
            torchaudio.save(tmp_audio, wav, sample_rate)
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_video, "-i", tmp_audio, "-c:v", "copy", "-c:a", "aac", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            Path(tmp_video).unlink(missing_ok=True)
            Path(tmp_audio).unlink(missing_ok=True)

        _write(fr_a, out_a)
        _write(fr_b, out_b)
        print(f"Saved best A/B pair: {out_a} and {out_b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

