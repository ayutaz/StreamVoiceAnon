UV-based setup (Windows/Linux/macOS)

This guide shows how to set up and run StreamVoiceAnon using uv.

- Requires Python 3.10–3.11 (recommended) and a CUDA-capable GPU for real-time.
- Windows users: Triton for Windows is required for low-latency `torch.compile`.

1) Install uv
- Windows (PowerShell):
  - iwr -useb https://astral.sh/uv/install.ps1 | iex
- macOS/Linux:
  - curl -LsSf https://astral.sh/uv/install.sh | sh

2) Create a virtual environment and activate
- uv venv .venv
- Windows PowerShell: .\.venv\Scripts\Activate.ps1
- macOS/Linux: source .venv/bin/activate

3) Install project dependencies with uv
- Base (offline inference):
  - uv sync
- Optional GUI/VAD extras (for real-time GUI):
  - uv sync --extra gui

4) Install PyTorch (choose one)
- CPU only (simplest):
  - uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
- CUDA 12.1 (stable 2.4 wheels):
  - uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
- CUDA 12.6 (nightly; fastest but bleeding edge):
  - uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

5) Windows only: Triton (for Inductor compile)
- uv pip install triton-windows==3.2.0.post13

6) Download pretrained checkpoints
- Option A (hf new CLI):
  - uv tool run hf download Plachta/StreamVoiceAnon --local-dir pretrained_checkpoints/
- Option B (classic):
  - uv tool run huggingface-cli download Plachta/StreamVoiceAnon --repo-type model --local-dir pretrained_checkpoints/

7) Quick tests
- Offline inference:
  - python evaluations/infer_arvc.py \
      --src_path ./test_waves/azuma_0.wav \
      --ref_path ./test_waves/trump_0.wav \
      --out_dir ./audio_outputs/ \
      --delay 2 \
      --compile

- Real-time GUI (microphone):
  - python evaluations/real-time-gui.py \
      --config_path configs/config_firefly_arvcasr_8192_delay0_8.yaml \
      --checkpoint_path pretrained_checkpoints/dual_ar_delay_0_8.pth

Notes
- If GUI audio is choppy, lower `block_frame` in the GUI and ensure CUDA + Triton are installed.
- On macOS (MPS), performance is lower; CPU-only may be slow for real-time.
- If `uv sync` installs CPU PyTorch but you have a GPU, rerun step 4 to install the correct CUDA wheels.
