"""
Main execution script for the Kanji generation project
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command with proper error handling"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"COMMAND: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout)
    else:
        print("✗ FAILED")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Kanji Generation Pipeline")
    parser.add_argument("--stage", choices=["data", "train", "inference", "all"], 
                       default="all", help="Which stage to run")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    
    args = parser.parse_args()
    
    # Get the Python executable path
    python_exe = r"C:/Users/gupta/OneDrive/Desktop/SAKANA AI/PART 3/.venv/Scripts/python.exe"
    
    # Stage 1: Data Preparation
    if args.stage in ["data", "all"]:
        print("🔄 Starting data preparation...")
        cmd = f'"{python_exe}" src/data_preparation.py'
        if not run_command(cmd, "Data preparation"):
            print("❌ Data preparation failed!")
            return
        print("✅ Data preparation completed!")
    
    # Stage 2: Training
    if args.stage in ["train", "all"]:
        print("🔄 Starting model training...")
        
        # Check if dataset exists
        if not os.path.exists("data/kanji_dataset.json"):
            print("❌ Dataset not found! Please run data preparation first.")
            return
        
        # Prepare training command
        cmd = f'"{python_exe}" src/train.py'
        cmd += f' --dataset_path "data/kanji_dataset.json"'
        cmd += f' --output_dir "models/kanji-sd"'
        cmd += f' --resolution {args.resolution}'
        cmd += f' --train_batch_size {args.batch_size}'
        cmd += f' --num_train_epochs {args.epochs}'
        cmd += f' --learning_rate 1e-5'
        cmd += f' --lr_scheduler "constant"'
        cmd += f' --lr_warmup_steps 0'
        cmd += f' --mixed_precision "fp16"' if args.gpu else ""
        cmd += f' --enable_xformers_memory_efficient_attention' if args.gpu else ""
        cmd += f' --dataloader_num_workers 0'
        cmd += f' --save_steps 500'
        cmd += f' --checkpointing_steps 500'
        cmd += f' --logging_dir "logs"'
        
        if not run_command(cmd, "Model training"):
            print("❌ Training failed!")
            return
        print("✅ Training completed!")
    
    # Stage 3: Inference
    if args.stage in ["inference", "all"]:
        print("🔄 Starting inference...")
        
        # Check if model exists
        if not os.path.exists("models/kanji-sd"):
            print("❌ Trained model not found! Please run training first.")
            return
        
        # Test prompts including novel concepts
        test_prompts = [
            "love", "peace", "water", "fire", "mountain",  # Traditional concepts
            "internet", "computer", "smartphone", "wifi", "email",  # Technology
            "artificial intelligence", "machine learning", "robot", "algorithm",  # AI
            "climate change", "global warming", "renewable energy", "solar power",  # Environment
            "social media", "youtube", "instagram", "facebook", "tiktok",  # Social media
            "elon musk", "tesla", "spacex", "mars", "rocket",  # Modern figures/companies
            "cryptocurrency", "bitcoin", "blockchain", "nft",  # Crypto
            "pandemic", "vaccine", "mask", "quarantine",  # Recent events
            "virtual reality", "metaverse", "gaming", "streaming",  # Digital concepts
            "baby robot", "armed fish", "language model", "space cat"  # Abstract/fun concepts
        ]
        
        cmd = f'"{python_exe}" src/inference.py'
        cmd += f' --model_path "models/kanji-sd"'
        cmd += f' --output_dir "outputs/generated"'
        cmd += f' --num_inference_steps 50'
        cmd += f' --guidance_scale 7.5'
        cmd += f' --num_images_per_prompt 4'
        cmd += f' --seed 42'
        
        # Add prompts to command
        for prompt in test_prompts:
            cmd += f' --prompts "{prompt}"'
        
        if not run_command(cmd, "Inference"):
            print("❌ Inference failed!")
            return
        print("✅ Inference completed!")
    
    print("\n🎉 Pipeline completed successfully!")
    print("\nGenerated files:")
    print("📁 data/ - Dataset and raw data files")
    print("📁 models/kanji-sd/ - Trained Stable Diffusion model")
    print("📁 outputs/generated/ - Generated Kanji images")
    print("📁 logs/ - Training logs and tensorboard files")

if __name__ == "__main__":
    main()
