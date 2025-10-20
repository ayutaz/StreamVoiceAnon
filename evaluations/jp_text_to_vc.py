import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import torch

# Reuse existing inference wrapper
sys.path.append(str(Path(__file__).resolve().parent.parent))
from evaluations.infer_arvc import InferenceWrapper  # noqa: E402


def synthesize_japanese_tts(text: str, out_wav: Path, voice: str | None = None, sample_rate: int = 44100) -> None:
    """Synthesize Japanese WAV using Windows System.Speech via PowerShell.

    Args:
        text: Japanese text to speak.
        out_wav: Output WAV path.
        voice: Optional specific voice name (exact match). If None, the first ja-* voice is used.
        sample_rate: WAV sample rate.
    """
    if os.name != "nt":
        raise RuntimeError("Japanese TTS helper supports Windows only (uses System.Speech).")

    out_wav = out_wav.resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        text_file = td_path / "text.txt"
        ps1_file = td_path / "run_tts.ps1"

        # Write text (UTF-8)
        text_file.write_text(text, encoding="utf-8")

        # Prepare PowerShell script
        # - Prefer a ja-* voice; allow explicit selection by name when provided
        # - Generate 44.1kHz/16bit/Mono WAV to match repo defaults
        ps_script = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $voiceName = {('"' + voice.replace('"','`"') + '"') if voice else '$null'}
    if ($voiceName) {{ $s.SelectVoice($voiceName) }}
    else {{
        $ja = $s.GetInstalledVoices() | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'ja-*' }} | Select-Object -First 1
        if (-not $ja) {{ throw '日本語音声(ja-*)が見つかりません' }}
        $s.SelectVoice($ja.VoiceInfo.Name)
    }}
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({sample_rate}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $s.SetOutputToWaveFile('{str(out_wav)}', $fmt)
    $text = Get-Content -Raw -Encoding UTF8 '{str(text_file)}'
    $s.Speak($text)
}} finally {{
    $s.SetOutputToDefaultAudioDevice()
    $s.Dispose()
}}
"""
        ps1_file.write_text(ps_script, encoding="utf-8")

        # Run PowerShell
        subprocess.run([
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(ps1_file),
        ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Japanese speech from text and run VC")
    parser.add_argument("--text", type=str, help="Japanese text to synthesize. If omitted, --src_path must be provided.", default=None)
    parser.add_argument("--src_path", type=str, help="Optional direct path to source WAV (skip TTS)", default=None)
    parser.add_argument("--ref_path", type=str, required=True, help="Reference speaker WAV path")
    parser.add_argument("--out_dir", type=str, default="./audio_outputs/", help="Output directory")
    parser.add_argument("--voice", type=str, default=None, help="Windows voice name to use for TTS (exact match)")

    # Model/config
    parser.add_argument("--config_path", type=str, default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_checkpoints/dual_ar_delay_0_8.pth")
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--compile", action="store_true")

    # Sampling controls (defaults tuned for content preservation)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If text provided, synthesize to WAV; otherwise, require --src_path
    if args.text:
        tts_wav = out_dir / "tts_src.wav"
        synthesize_japanese_tts(args.text, tts_wav, voice=args.voice)
        src_path = str(tts_wav)
    else:
        if not args.src_path:
            parser.error("Either --text or --src_path must be provided.")
        src_path = args.src_path

    # Modest performance hint (TF32) on Ampere+ GPUs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    compile_enabled = bool(args.compile and torch.cuda.is_available())

    wrapper = InferenceWrapper(
        args.config_path,
        args.checkpoint_path,
        compile_ar=compile_enabled,
        compile_decoder=False,
        compile_encoder=False,
    )

    # Run offline VC
    _ = wrapper.infer(
        src_path=src_path,
        ref_path=args.ref_path,
        out_dir=str(out_dir),
        delay=args.delay,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()

