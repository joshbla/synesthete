import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import load_config, get_device
from vae import VAE
from audio_gen import AudioGenerator
from audio_features import compute_audio_timeline
from visualizers.debug import DebugPulseGlobal, DebugBandsBars


def _shuffle_T(x: torch.Tensor, *, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(int(x.shape[0]), generator=g)
    return x[perm]


def _write_video_with_audio(frames: torch.Tensor, waveform: torch.Tensor, *, fps: int, sample_rate: int, path: str) -> None:
    from torchcodec.encoders import VideoEncoder
    import torchaudio
    import subprocess

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    video_uint8 = (frames.clamp(0, 1).cpu() * 255).byte()
    enc = VideoEncoder(video_uint8, frame_rate=fps)
    tmp_video = path.replace(".mp4", "_tmp.mp4")
    tmp_audio = path.replace(".mp4", "_tmp.wav")
    enc.to_file(tmp_video)
    torchaudio.save(tmp_audio, waveform.cpu(), sample_rate)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_video, "-i", tmp_audio, "-c:v", "copy", "-c:a", "aac", path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(tmp_video).unlink(missing_ok=True)
    Path(tmp_audio).unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Check whether VAE encode->decode preserves audio-reactive differences.")
    p.add_argument("--override-config", type=str, default=None)
    p.add_argument("--kind", choices=["pulse", "bars"], default="pulse")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--save-videos", action="store_true", help="Write gt/vae A/B videos to outputs/vae_check/")
    args = p.parse_args()

    cfg = load_config(override_path=args.override_config) if args.override_config else load_config()
    device = get_device()

    sample_rate = int(cfg.get("data", {}).get("sample_rate", 16000))
    fps = int(cfg.get("data", {}).get("fps", 15))
    duration = float(args.duration if args.duration is not None else cfg.get("data", {}).get("duration", 6.0))
    height = int(cfg.get("data", {}).get("height", 128))
    width = int(cfg.get("data", {}).get("width", 128))
    num_frames = int(cfg.get("model", {}).get("num_frames", int(duration * fps)))

    n_fft = int(cfg.get("diffusion", {}).get("audio_feature_n_fft", 512))
    num_bands = int(cfg.get("diffusion", {}).get("audio_feature_num_bands", 8))
    include_abs_rms = bool(cfg.get("diffusion", {}).get("audio_feature_include_abs_rms", False))
    include_onset = bool(cfg.get("diffusion", {}).get("audio_feature_include_onset", False))
    include_flux_bands = bool(cfg.get("diffusion", {}).get("audio_feature_include_flux_bands", False))

    # Make an audio clip with explicit silences/dropouts (most human-auditable)
    gen = AudioGenerator(sample_rate=sample_rate)
    waveform = gen.generate_sequence_dropouts(duration=duration, bpm=120, full_silence=True)  # (1, N)

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
    )
    feats_shuf = _shuffle_T(feats, seed=args.shuffle_seed)

    viz = DebugPulseGlobal() if args.kind == "pulse" else DebugBandsBars()
    frames_a = viz.render(waveform, fps=fps, height=height, width=width, sample_rate=sample_rate, audio_feats=feats).clamp(0, 1)
    frames_b = viz.render(waveform, fps=fps, height=height, width=width, sample_rate=sample_rate, audio_feats=feats_shuf).clamp(0, 1)

    # Ground-truth reactivity (pixel space)
    gt_l1 = torch.mean(torch.abs(frames_a - frames_b)).item()
    gt_rel = float(gt_l1 / (torch.mean(torch.abs(frames_a)) + 1e-8).item())

    # VAE encode->decode
    latent_dim = int(cfg.get("model", {}).get("latent_dim", 256))
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load("vae_checkpoints/vae_latest.pth", map_location=device))
    vae.eval()
    with torch.no_grad():
        mu_a, _ = vae.encode(frames_a.to(device))
        mu_b, _ = vae.encode(frames_b.to(device))
        dec_a = vae.decode(mu_a).clamp(0, 1).cpu()
        dec_b = vae.decode(mu_b).clamp(0, 1).cpu()

    vae_l1 = torch.mean(torch.abs(dec_a - dec_b)).item()
    vae_rel = float(vae_l1 / (torch.mean(torch.abs(dec_a)) + 1e-8).item())
    recon_l1 = torch.mean(torch.abs(dec_a - frames_a.cpu())).item()

    print("VAE reactivity check (debug renderer A vs shuffled-Audio-feats B):")
    print(f"- gt_frame_rel_l1:  {gt_rel:.6f}")
    print(f"- vae_frame_rel_l1: {vae_rel:.6f}")
    print(f"- vae_recon_l1:     {recon_l1:.6f}  (how much VAE changes the target frames)")

    if args.save_videos:
        out_dir = Path("outputs/vae_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{args.kind}_{int(duration)}s_seed{int(args.shuffle_seed)}"
        _write_video_with_audio(frames_a, waveform, fps=fps, sample_rate=sample_rate, path=str(out_dir / f"{base}_gt_normal.mp4"))
        _write_video_with_audio(frames_b, waveform, fps=fps, sample_rate=sample_rate, path=str(out_dir / f"{base}_gt_shuffled.mp4"))
        _write_video_with_audio(dec_a, waveform, fps=fps, sample_rate=sample_rate, path=str(out_dir / f"{base}_vae_normal.mp4"))
        _write_video_with_audio(dec_b, waveform, fps=fps, sample_rate=sample_rate, path=str(out_dir / f"{base}_vae_shuffled.mp4"))
        print(f"Wrote videos under outputs/vae_check/ with prefix {base}_*")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

