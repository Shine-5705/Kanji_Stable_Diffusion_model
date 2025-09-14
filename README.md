# Kanji Stable Diffusion Training Project

This project fine-tunes Stable Diffusion to generate novel Kanji characters from English definitions, following the research approach described in the referenced Twitter thread.

## Overview

The goal is to train a Stable Diffusion model that can generate new "Kanji" characters for modern concepts like "YouTube", "Elon Musk", "artificial intelligence", etc., by learning from a dataset of traditional Kanji characters and their English meanings.

## Project Structure

```
PART 3/
├── data/                          # Dataset storage (created by setup)
│   ├── raw/                       # Raw downloaded files
│   ├── images/                    # Processed Kanji images
│   └── kanji_dataset.json         # Final dataset
├── src/                           # Source code
│   ├── data_preparation.py        # Download and process KANJIDIC2 + KanjiVG data
│   ├── dataset.py                 # PyTorch dataset classes
│   ├── train.py                   # Stable Diffusion fine-tuning script
│   └── inference.py               # Generate novel Kanji images
├── models/                        # Trained model storage
│   └── kanji-sd/                  # Fine-tuned Stable Diffusion model
├── outputs/                       # Generated images
│   └── generated/                 # Novel Kanji generations
├── logs/                          # Training logs and tensorboard
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file (excludes data/)
└── README.md                      # This file
```

**⚠️ Important**: The `data/` folder is **not included** in this repository due to size constraints (~700MB). You must run the setup process to download and create the dataset before training.

## Quick Start

### 1. Environment Setup

The virtual environment is already configured. Install dependencies:

```bash
# The Python environment is already set up
# Dependencies should already be installed
```

### 2. Dataset Setup

**Important**: The `data/` folder is not included in the repository due to size constraints. You need to create it and download the datasets:

#### Option A: Automatic Setup (Recommended)
```bash
# This will create the data folder structure and download all required datasets
python src/data_preparation.py
```

#### Option B: Manual Setup
```bash
# 1. Create the data folder structure
mkdir data
mkdir data/raw
mkdir data/images

# 2. Download KANJIDIC2 dataset
# Download from: https://www.edrdg.org/kanjidic/kanjidic2.xml.gz
# Extract and place kanjidic2.xml in data/raw/

# 3. Download KanjiVG dataset  
# Download from: https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz
# Extract and place all SVG files in data/raw/kanjivg/

# 4. Process the datasets
python src/data_preparation.py
```

**Expected folder structure after setup:**
```
data/
├── raw/                           # Raw downloaded files (~500MB)
│   ├── kanjidic2.xml             # Kanji dictionary with English meanings
│   └── kanjivg/                  # Folder containing ~6,000 SVG files
│       ├── 04e00.svg
│       ├── 04e01.svg
│       └── ...
├── images/                        # Processed PNG images (~200MB)
│   ├── kanji_04e00.png
│   ├── kanji_04e01.png
│   └── ...
└── kanji_dataset.json            # Final training dataset (~5MB)
```

**Note**: The complete dataset is approximately **~700MB** and contains:
- **KANJIDIC2**: ~13,000 Kanji characters with English meanings
- **KanjiVG**: ~6,400 SVG stroke data files
- **Processed**: ~6,400 PNG images (256×256 pixels) ready for training

### 3. Run the Complete Pipeline

To run everything (data preparation, training, and inference):

```bash
python main.py --stage all --gpu --epochs 10 --batch_size 4
```

Or run individual stages:

```bash
# Data preparation only
python main.py --stage data

# Training only (after data is prepared)
python main.py --stage train --gpu --epochs 10

# Inference only (after model is trained)
python main.py --stage inference
```

### 3. Manual Execution

You can also run each component individually:

```bash
# 1. Prepare dataset
python src/data_preparation.py

# 2. Train the model
python src/train.py --dataset_path data/kanji_dataset.json --output_dir models/kanji-sd --num_train_epochs 10 --mixed_precision fp16 --enable_xformers_memory_efficient_attention

# 3. Generate novel Kanji
python src/inference.py --model_path models/kanji-sd --prompts "youtube" "artificial intelligence" "elon musk"
```

## Data Sources

- **KANJIDIC2**: English meanings for ~10,000+ Kanji characters
  - URL: https://www.edrdg.org/kanjidic/kanjidic2.xml.gz
  
- **KanjiVG**: SVG stroke data for Kanji characters
  - URL: https://github.com/KanjiVG/kanjivg/releases/download/r20220427/kanjivg-20220427.xml.gz

## Training Details

### Model Architecture
- Base model: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)
- Fine-tuning approach: UNet fine-tuning with frozen VAE and text encoder
- Image resolution: 256x256 pixels (for faster training)
- Text encoder: CLIP (unchanged, allows interpolation for novel concepts)

### Training Configuration
- **Epochs**: 10 (adjustable)
- **Batch size**: 4 (adjust based on GPU memory)
- **Learning rate**: 1e-5
- **Mixed precision**: FP16 (for GPU acceleration)
- **Optimizer**: AdamW
- **Scheduler**: Constant learning rate

### Hardware Requirements
- **GPU**: Recommended for training (CUDA-compatible)
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB for dataset + model checkpoints
- **Training time**: ~2-4 hours on modern GPU

## Generated Examples

The model will generate Kanji for concepts like:

### Traditional Concepts
- love, peace, water, fire, mountain

### Technology
- internet, computer, smartphone, wifi, email

### AI & Modern Tech
- artificial intelligence, machine learning, robot, algorithm

### Social Media & Companies
- youtube, instagram, facebook, elon musk, tesla

### Environmental
- climate change, renewable energy, solar power

### Abstract/Fun
- baby robot, armed fish, language model, space cat

## Output Analysis

Generated images will be saved in `outputs/generated/` with descriptive filenames. Analyze results for:

1. **Success cases**: Clear, Kanji-like structures for novel concepts
2. **Failure cases**: Unclear or non-Kanji-like outputs
3. **Interesting interpolations**: How the model combines concepts
4. **Cultural relevance**: Whether generated "Kanji" feel authentic

## Evaluation Metrics

While subjective, consider:
- **Structural similarity** to real Kanji (stroke-like patterns)
- **Concept representation** (does it visually relate to the English meaning?)
- **Novelty** (different from existing Kanji)
- **Artistic quality** (clean, well-formed characters)

## Technical Implementation Notes

### Data Processing
- SVG files converted to 256x256 PNG images
- Black strokes (#000000) on white background
- Stroke order numbers removed
- Multiple English meanings per Kanji combined

### Model Modifications
- Only UNet parameters are fine-tuned
- Text encoder remains frozen (enables generalization to novel concepts)
- VAE remains frozen (maintains image quality)
- DDPM noise scheduler for training

### Memory Optimizations
- XFormers attention (if available)
- Mixed precision training (FP16)
- Gradient checkpointing
- Batch size tuning based on GPU memory

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--train_batch_size 2`
   - Reduce resolution: `--resolution 128`

2. **Slow training without GPU**
   - Remove `--gpu` flag and `--mixed_precision`
   - Consider using Google Colab or cloud GPU

3. **Dataset download fails**
   - Check internet connection
   - Manual download and place in `data/raw/` folder
   - Ensure you have ~1GB free disk space

4. **Data folder missing or empty**
   ```bash
   # Run this to set up the complete data structure:
   python src/data_preparation.py
   
   # Verify the setup:
   ls data/                    # Should show: raw/, images/, kanji_dataset.json
   ls data/raw/               # Should show: kanjidic2.xml, kanjivg/
   ls data/images/ | wc -l    # Should show: ~6400 PNG files
   ```

5. **Permission errors when creating folders**
   - Run terminal as administrator (Windows)
   - Check write permissions in the project directory

6. **Poor generation quality**
   - Train for more epochs: `--epochs 20`
   - Adjust guidance scale in inference: `--guidance_scale 10.0`

## References

1. Original research thread: https://twitter.com/hardmaru/status/1611237067589095425
2. KANJIDIC2 project: https://www.edrdg.org/wiki/index.php/KANJIDIC_Project
3. KanjiVG project: https://kanjivg.tagaini.net/
4. Stable Diffusion: https://github.com/CompVis/stable-diffusion
5. Hugging Face Diffusers: https://huggingface.co/docs/diffusers/
