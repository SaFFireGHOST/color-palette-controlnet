
#  Color Palette–Conditioned Image Generation using ControlNet

This repository contains the complete codebase for the project **“Applying ControlNet for Color Palette–Conditioned Art Generation”**.
The project explores explicit and controllable color composition in diffusion-based image generation using a novel **palette-based conditioning signal** with ControlNet.

---

##  Repository Structure

```text
.
├── color-palette-generation.ipynb   # Dataset generation & preprocessing
├── controlnet-training.ipynb         # ControlNet training (Diffusers)
├── color-palette-inference.ipynb     # Inference, ablations, evaluation
├── sample_outputs/                   # Generated samples and figures (optional)
└── README.md
```

---

## 🧩 Project Components

### 1. Dataset Generation and Preprocessing

**File:** `color-palette-generation.ipynb`

This notebook constructs a ControlNet-compatible dataset from the **ArtCap** dataset.

**Steps performed:**

* Load artwork images and captions
* Resize images to 512 × 512
* Extract **5 dominant colors** using K-Means clustering
* Encode palettes as **horizontal stripe conditioning images**
* Save paired `(image, conditioning_image, prompt)` samples
* Export metadata in Hugging Face–compatible format

**Outputs:**

* `images/` – target images
* `conditions/` – palette conditioning images
* `prompts/` – text captions
* `metadata.json`

---

### 2. ControlNet Training

**File:** `controlnet-training.ipynb`

This notebook trains a ControlNet model on top of **Stable Diffusion v1.5** using the Hugging Face Diffusers framework.

**Training setup:**

* Base model: `runwayml/stable-diffusion-v1-5`
* ControlNet initialized from pretrained weights
* Resolution: 512 × 512
* Mixed precision (FP16)
* Gradient accumulation and checkpointing
* 50% prompt dropout to encourage reliance on palette conditioning

**Experiments enabled:**

* Conditioning scale ablation
* Prompt influence analysis
* Guidance scale (CFG) analysis
* Robustness to noisy conditioning images

**Outputs:**

* Trained ControlNet checkpoints
* Option to upload trained weights to Hugging Face Hub

---

### 3. Inference and Evaluation

**File:** `color-palette-inference.ipynb`

This notebook demonstrates image generation and evaluation using the trained ControlNet.

**Features:**

* Load trained ControlNet and Stable Diffusion pipeline
* Generate images using palette and text conditioning
* Perform qualitative comparisons
* Run ablation studies:

  * Conditioning strength
  * Prompt complexity
  * Guidance scale (CFG)
  * Noise robustness
* Compute a **Palette Adherence Metric** for quantitative evaluation
* Save visual grids and evaluation plots

---

##  Reproducibility Instructions

### Environment Setup

Recommended: **Python 3.10 with CUDA 11.8**

Install dependencies:

```bash
pip install torch torchvision diffusers transformers accelerate datasets
pip install scikit-learn pillow opencv-python matplotlib tqdm
```

(Optional but recommended)

```bash
pip install xformers
```

### Execution Order

1. Run `color-palette-generation.ipynb` to generate the dataset
2. Run `controlnet-training.ipynb` to train the ControlNet
3. Run `color-palette-inference.ipynb` to perform inference and evaluation

Random seeds are fixed where applicable to improve reproducibility.

---

##  Path and Environment Notes

This project was primarily developed and executed in a **Kaggle notebook environment**. As a result, several dataset and output paths follow Kaggle’s filesystem conventions.

If running the code on a different platform (e.g., Google Colab or a local machine), **please modify dataset and output paths accordingly**. The code itself is platform-agnostic and does not rely on Kaggle-specific APIs.

---

##  Sample Outputs

The `sample_outputs/` directory (or figures included in the report) contains:

* Palette-conditioned image generations
* Prompt influence comparisons
* Conditioning scale ablations
* Noise robustness evaluations
* Failure cases under contradictory prompts

These outputs directly correspond to figures presented in the final report.

---

##  Notes

* This repository is intended for **academic and educational use**
* Training requires a GPU-enabled environment
* Datasets and pretrained models are sourced from publicly available repositories

