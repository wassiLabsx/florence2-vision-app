# Florence-2 Vision App

A small local web app that runs Microsoft's [Florence-2](https://huggingface.co/microsoft/Florence-2-base) vision model on your own machine. Upload an image, pick a task, and get a caption or detected objects. Built with Streamlit and Hugging Face Transformers.

## Features

- Upload an image (JPG, PNG or WebP)
- Choose a vision task:
  - `<CAPTION>`: short caption
  - `<DETAILED_CAPTION>`: longer description
  - `<MORE_DETAILED_CAPTION>`: very detailed description
  - `<OD>`: object detection, with bounding boxes and labels drawn on the image
- Runs locally on CUDA GPU, Apple Silicon (MPS) or CPU, chosen automatically

## Requirements

- Python 3.10 or newer
- About 2 GB of free disk space (model weights and dependencies)
- An internet connection on the first run, to download the model

## Installation

```powershell
# 1. Clone the repository
git clone https://github.com/wassiLabsx/florence2-vision-app.git
cd florence2-vision-app

# 2. Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate` instead.

If you want GPU acceleration, install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) before step 3.

## Usage

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (usually at `http://localhost:8501`). The first launch downloads the Florence-2 model, which can take a few minutes.

1. Upload an image.
2. Select a task from the dropdown.
3. Click **Run Inference**.

## Important notes

- **Pin `transformers==4.49.0`.** Florence-2 uses custom model code (`trust_remote_code=True`) that is sensitive to the Transformers version. Newer versions can produce garbled output or errors.
- **Precision.** The app uses `float32` on all devices. Switching to `float16` on a GPU saves memory, but test it first, since it can degrade output on some setups.
- **Trust remote code.** The model loads custom code from the Hugging Face Hub. Only run it if you are comfortable with that.

## Troubleshooting

| Problem | Likely fix |
| --- | --- |
| Gibberish or broken output | Check `pip show transformers` reports `4.49.0`, and do not patch Transformers internals |
| `ModuleNotFoundError: einops` or `timm` | Run `pip install -r requirements.txt` again |
| Very slow generation | Normal on CPU. Use a CUDA GPU if available |
| Out of memory | Close other GPU programs, or use a smaller image |

## Project structure

```
florence2-vision-app/
├── app.py             # Streamlit application
├── requirements.txt   # Python dependencies
└── .gitignore
```

## Acknowledgements

- [Florence-2](https://huggingface.co/microsoft/Florence-2-base) by Microsoft
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
