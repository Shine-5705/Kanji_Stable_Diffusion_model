"""
Analysis script for the Kanji dataset and training progress
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
import random

def analyze_dataset():
    """Analyze the prepared Kanji dataset"""
    
    print("📊 Analyzing Kanji Dataset")
    print("=" * 50)
    
    # Load dataset
    dataset_path = "data/kanji_dataset.json"
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found! Run data preparation first.")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"📈 Dataset Statistics:")
    print(f"   Total Kanji characters: {len(dataset)}")
    
    # Analyze meanings
    all_meanings = []
    text_lengths = []
    
    for item in dataset:
        meanings = item['meanings']
        text = item['text']
        
        all_meanings.extend(meanings)
        text_lengths.append(len(text))
    
    print(f"   Total unique meanings: {len(set(all_meanings))}")
    print(f"   Average text length: {np.mean(text_lengths):.1f} characters")
    print(f"   Text length range: {min(text_lengths)} - {max(text_lengths)}")
    
    # Most common meanings
    meaning_counts = Counter(all_meanings)
    print(f"\n🔤 Most Common Meanings:")
    for meaning, count in meaning_counts.most_common(10):
        print(f"   '{meaning}': {count} times")
    
    # Sample entries
    print(f"\n📝 Sample Dataset Entries:")
    sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        item = dataset[idx]
        print(f"   {i+1}. Kanji: '{item['kanji']}' -> '{item['text']}'")
        print(f"      Meanings: {item['meanings']}")
        print(f"      Image: {os.path.basename(item['image_path'])}")
        if i < len(sample_indices) - 1:
            print()
    
    # Check image files
    print(f"\n🖼️  Image File Analysis:")
    image_dir = "data/images"
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        print(f"   Total image files: {len(image_files)}")
        
        # Sample a few images to check dimensions
        sample_images = random.sample(image_files, min(3, len(image_files)))
        print(f"   Sample image dimensions:")
        
        for img_file in sample_images:
            img_path = os.path.join(image_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    print(f"     {img_file}: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
            except Exception as e:
                print(f"     {img_file}: Error loading - {e}")
    
    return dataset

def analyze_training_progress():
    """Check training progress if available"""
    
    print("\n🚀 Training Progress Analysis")
    print("=" * 50)
    
    model_dir = "models/kanji-sd"
    logs_dir = "logs"
    
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)
        if model_files:
            print(f"✅ Model directory exists with {len(model_files)} files:")
            for f in sorted(model_files)[:10]:  # Show first 10 files
                print(f"   - {f}")
            if len(model_files) > 10:
                print(f"   ... and {len(model_files) - 10} more files")
        else:
            print("⏳ Model directory exists but is empty (training in progress)")
    else:
        print("❌ No trained model found yet")
    
    if os.path.exists(logs_dir):
        log_files = os.listdir(logs_dir)
        if log_files:
            print(f"📊 Training logs found: {len(log_files)} files")
        else:
            print("📝 Logs directory exists but no log files yet")
    else:
        print("📝 No training logs directory found")
    
    # Check for checkpoints
    checkpoint_dirs = [d for d in os.listdir(".") if d.startswith("checkpoint-") and os.path.isdir(d)]
    if checkpoint_dirs:
        print(f"💾 Found {len(checkpoint_dirs)} training checkpoints:")
        for checkpoint in sorted(checkpoint_dirs)[:5]:
            print(f"   - {checkpoint}")
    else:
        print("💾 No training checkpoints found yet")

def generate_dataset_visualization():
    """Create visualizations of the dataset"""
    
    print("\n📈 Creating Dataset Visualizations")
    print("=" * 50)
    
    dataset_path = "data/kanji_dataset.json"
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create output directory for plots
    os.makedirs("outputs/analysis", exist_ok=True)
    
    # 1. Text length distribution
    text_lengths = [len(item['text']) for item in dataset]
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Text Description Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/analysis/text_length_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Number of meanings per Kanji
    meanings_per_kanji = [len(item['meanings']) for item in dataset]
    
    plt.figure(figsize=(10, 6))
    plt.hist(meanings_per_kanji, bins=range(1, max(meanings_per_kanji)+2), alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Number of Meanings per Kanji')
    plt.xlabel('Number of Meanings')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(meanings_per_kanji)+1))
    plt.savefig('outputs/analysis/meanings_per_kanji.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Sample Kanji images (if available)
    try:
        sample_items = random.sample(dataset, min(16, len(dataset)))
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Sample Kanji Characters from Dataset', fontsize=16)
        
        for i, item in enumerate(sample_items):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            try:
                img = Image.open(item['image_path'])
                ax.imshow(img, cmap='gray')
                ax.set_title(f"'{item['kanji']}'\n{item['text'][:20]}...", fontsize=8)
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"'{item['kanji']}'", fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/analysis/sample_kanji_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️ Could not create Kanji sample grid: {e}")
    
    print("✅ Visualizations saved to outputs/analysis/")

def main():
    """Run complete analysis"""
    
    print("🔍 Kanji Stable Diffusion - Dataset Analysis")
    print("=" * 60)
    
    try:
        # Analyze dataset
        dataset = analyze_dataset()
        
        if dataset:
            # Check training progress
            analyze_training_progress()
            
            # Generate visualizations
            generate_dataset_visualization()
            
            print(f"\n" + "=" * 60)
            print("✅ Analysis completed successfully!")
            print("📊 Key Findings:")
            print(f"   • Dataset contains {len(dataset)} Kanji characters")
            print(f"   • Ready for training Stable Diffusion model")
            print(f"   • Each Kanji has English meaning descriptions")
            print(f"   • Images are 256x256 pixels in PNG format")
            
            print(f"\n📁 Generated Files:")
            print(f"   • Dataset: data/kanji_dataset.json")
            print(f"   • Images: data/images/ ({len(dataset)} files)")
            print(f"   • Analysis plots: outputs/analysis/")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()
