import argparse
import sys
import subprocess
from pathlib import Path


DEFAULT_TEXT = (
    "若き日の反逆ゆえに、宇宙の中央を追放されて、惑星、地球にやってきた主人公、"
    "ベルゼバブが、宇宙船カルナークのなかで、孫に語る壮大な物語。"
)
DEFAULT_REF_PATH = "VOICEACTRESS100_094.wav"
DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent / "audio_outputs")


def main():
    parser = argparse.ArgumentParser(description="Japanese text -> TTS -> VC wrapper")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--ref_path", type=str, default=DEFAULT_REF_PATH)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--config_path", type=str, default="configs/config_firefly_arvcasr_8192_delay0_8.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="pretrained_checkpoints/dual_ar_delay_0_8.pth")
    # sampling defaults favor content preservation
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path("evaluations") / "jp_text_to_vc.py"),
        "--text", args.text,
        "--ref_path", args.ref_path,
        "--out_dir", args.out_dir,
        "--delay", str(args.delay),
        "--config_path", args.config_path,
        "--checkpoint_path", args.checkpoint_path,
        "--top_p", str(args.top_p),
        "--temperature", str(args.temperature),
        "--repetition_penalty", str(args.repetition_penalty),
        "--compile",
    ]

    print("Running:")
    print(" ".join(f'"{c}"' if (" " in c and not c.startswith("-")) else c for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
