# Qwen2.5-1.5B Fine-Tuning Guide: Electrical Engineering & Signals and Systems

This folder contains the complete, self-contained training suite to fine-tune **Qwen2.5-1.5B-Instruct** into a specialized spoken-mathematics-to-LaTeX model and export it to a quantized GGUF file for execution on the **Jetson Orin Nano**.

---

## Folder Structure

```
training/
├── data/
│   ├── train.jsonl.gz           # 10,467 unique training samples (0.83 MB compressed, 12.44 MB raw)
│   └── val.jsonl.gz             # 1,179 unique validation samples (96 KB compressed, 1.40 MB raw, 0% overlap)
├── train_qwen_signals_colab.ipynb # 1-Click Google Colab Notebook
├── train_qwen_signals.py        # Cross-platform training script (Windows / Linux)
├── requirements.txt             # Python dependencies
└── README.md                    # This guide
```

---

## Method 1: Train on Google Colab (Recommended, Free GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload Notebook** and select:
   `training/train_qwen_signals_colab.ipynb`
3. Go to **Runtime > Change runtime type**, select **T4 GPU**, and save.
4. Run all cells in order:
   - **Step 1**: Installs `transformers`, `peft`, `bitsandbytes`, `datasets`.
   - **Step 2**: Clones `https://github.com/aYush007-blip/MTP.git` and decompresses the dataset.
   - **Step 3**: Fine-tunes Qwen2.5-1.5B with 4-bit QLoRA (~15–25 minutes).
   - **Step 4**: Automatically merges the LoRA adapter and quantizes the model to `qwen2.5-1.5b-instruct-q4_k_m.gguf`.
   - **Step 5**: Automatically downloads `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.0 GB) to your browser downloads folder.

---

## Method 2: Train on Local Windows Machine (NVIDIA GPU)

### 1. Prerequisites (Windows)
- Python 3.10 or 3.11 installed.
- NVIDIA GPU with at least 6 GB of VRAM (RTX 3060, 3070, 3080, 4060, 4070, 4080, 4090, etc.).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

### 2. Setup Environment (PowerShell)
```powershell
# Clone the repository
git clone https://github.com/aYush007-blip/MTP.git
cd MTP\training

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA (check your CUDA version, e.g. cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install -r requirements.txt
```

### 3. Run Fine-Tuning
```powershell
python train_qwen_signals.py --qlora --epochs 3 --batch-size 4 --grad-accum 4 --max-length 1024
```
*Note: Uses assistant-only loss masking so gradients focus 100% on spoken-to-LaTeX output. If you have a 16GB+ GPU (RTX 3090, 4080, 4090), you can pass `--no-qlora` for full 16-bit LoRA training.*

After training completes, the merged HuggingFace model is saved in:
`training\output_qwen_signals\merged_model\`

---

## Method 3: Convert Merged Model to GGUF (Windows / Linux)

To convert the merged model into GGUF for the Jetson:

```bash
# 1. Clone llama.cpp if not already present
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# 2. Convert HuggingFace model to FP16 GGUF
python convert_hf_to_gguf.py ../output_qwen_signals/merged_model --outfile qwen2.5-1.5b-ee-fp16.gguf

# 3. Quantize FP16 to Q4_K_M (the fast edge format)
# Build llama-quantize with CMake (or download pre-built Windows release binaries from llama.cpp Releases):
cmake -B build -DGGML_CUDA=OFF
cmake --build build --target llama-quantize --config Release
./build/bin/llama-quantize qwen2.5-1.5b-ee-fp16.gguf qwen2.5-1.5b-instruct-q4_k_m.gguf Q4_K_M
```

---

## Transferring the New Model to Jetson Orin Nano

Once you have `qwen2.5-1.5b-instruct-q4_k_m.gguf`:

### Option A: Transfer via SCP over Wi-Fi/Ethernet
From your Windows/Colab terminal:
```bash
scp qwen2.5-1.5b-instruct-q4_k_m.gguf ayush02@<JETSON_IP>:/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### Option B: Transfer via USB Drive
1. Copy `qwen2.5-1.5b-instruct-q4_k_m.gguf` to a USB flash drive.
2. Plug the USB into the Jetson.
3. On the Jetson, copy the file:
   ```bash
   cp /media/ayush02/<USB_NAME>/qwen2.5-1.5b-instruct-q4_k_m.gguf /home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
   ```

### Verification on Jetson
The symlink `/mnt/thesis/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf` points directly to `/home/ayush02/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`.

Once replaced, restart the llama-server daemon or test with:
```bash
/mnt/thesis/headless_agent/.venv/bin/python3 -c "
import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:8080/completion', data=json.dumps({'prompt': '<|im_start|>user\nSpoken transcript: where x of s is the Laplace transform of x of t<|im_end|>\n<|im_start|>assistant\n', 'n_predict': 50}).encode('utf-8'), headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req).read().decode('utf-8'))['content'])
"
```
The model will now natively output academic LaTeX with zero fallback overhead!

