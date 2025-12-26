# Digital Human Project

This project integrates **CosyVoice** (for text-to-speech) and **DFA-NeRF** (for talking head generation) to create a digital human interface.

## Prerequisites

You need to have `conda` installed.

## Setup

### 1. Clone Repositories (Already done if you see the folders)

```bash
git clone https://github.com/ShunyuYao/DFA-NeRF.git
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
```

### 2. Setup CosyVoice Environment

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
# Install torch (adjust cuda version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd CosyVoice
pip install -r requirements.txt
# Download models (requires modelscope or huggingface)
# You can use the provided script in CosyVoice repo or download manually.
# For example:
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
```

### 3. Setup DFA-NeRF Environment

```bash
conda create -n adnerf python=3.8 -y
conda activate adnerf
cd DFA-NeRF
# Install dependencies (refer to their environment.yml or install manually)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt # If available, otherwise install:
pip install opencv-python tqdm scipy scikit-image tensorboardX
```

**Note:** DFA-NeRF requires pre-trained models and data. Please follow the instructions in `DFA-NeRF/README.md` to download the necessary weights and prepare the `obama` dataset (or your own).
Specifically, you need:
- `data_util/face_tracking/3DMM/` files
- `data_util/face_parsing/79999_iter.pth`
- `dataset/obama/` (trained model and config)

### 4. Setup Frontend Environment

```bash
conda create -n frontend python=3.10 -y
conda activate frontend
pip install streamlit
```

## Usage

1. Activate the frontend environment:
   ```bash
   conda activate frontend
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run frontend/app.py
   ```

## Directory Structure

- `CosyVoice/`: Text-to-Speech engine.
- `DFA-NeRF/`: Video generation engine.
- `frontend/`: Streamlit application and helper scripts.
