# StreamVoiceAnon
This repository contains the implementation of StreamVoiceAnon, a real-time voice anonymization / voice conversion model.
Relevant paper has been submitted to ICASSP 2026.  
Training code will be released after the paper is accepted.

### Installation (pip)
```bash
git clone https://github.com/Plachtaa/StreamVoiceAnon.git
cd StreamVoiceAnon
pip install -r requirements.txt
```

If running on Windows OS, please install the following:
```bash
pip install triton-windows==3.2.0.post13
```
Note that this is **compulsory** to run inference with RTF < 1.0

Full MacOS support is still under construction.

### Installation (uv)
If you prefer uv, follow docs/uv-setup.md for a faster, reproducible setup:

```bash
# 1) Create venv and install deps
uv venv .venv && source .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
uv sync                      # base deps
uv sync --extra gui          # add GUI/VAD extras (optional)

# 2) Install PyTorch (choose one)
# CPU only:
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1 (stable):
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6 (nightly):
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# 3) Windows only: Triton for Inductor
uv pip install triton-windows==3.2.0.post13
```

### Download Pretrained Models
```bash
hf download Plachta/StreamVoiceAnon --local-dir pretrained_checkpoints/
```

### Inference
Offline inference 
```bash
python evaluations/infer_arvc.py --src_path <path_to_audio> --ref_path <path_to_audio> --out_dir <path_to_output_directory> --delay 2 --compile
```
Simulated online inference
```bash
python evaluations/infer_arvc.py --src_path <path_to_audio> --ref_path <path_to_audio> --out_dir <path_to_output_directory> --delay 2 --compile --simulate_streaming --decode_chunk_frames 1
```
This simulates a chunk-by-chunk online inference with specified chunk size. `src_path` (source audio) has no length limit here. `ref_path` (reference audio) will be truncated to some maximum length (if longer than that limit)

Real-time inference
```bash
python evaluations/real-time-gui.py
```
This UI uses the same behavior as simulated online inference. It uses `--compile` by default, so please ensure you have installed triton (as previously stated) before using it.

### TODO
 - [ ] Full MacOS support
 - [ ] Release metrics for voice conversion & speaker anonymization
 - [ ] Release fine-tuning code
 - [ ] More to be added

### Acknowledgements
 - Co-author: https://github.com/paniquex
 - Computation resources: https://www.nscc.sg/
 - Real-time GUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
 - Speaker representations (1 of 2) https://huggingface.co/funasr/campplus
 - Speaker representations (2 of 2) https://github.com/SparkAudio/Spark-TTS
 - Speech acoustic codec https://huggingface.co/fishaudio/fish-speech-1.5
 - Idea: https://arxiv.org/html/2401.11053v1
