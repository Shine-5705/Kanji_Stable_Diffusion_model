"""
Inference script for generating novel Kanji with trained Stable Diffusion model
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kanji using trained Stable Diffusion model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "YouTube",
            "artificial intelligence", 
            "Elon Musk",
            "smartphone",
            "social media",
            "cryptocurrency",
            "virtual reality",
            "electric car"
        ],
        help="English concepts to generate Kanji for"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated_kanji",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Kanji Generation with Stable Diffusion")
    print(f"📁 Model path: {args.model_path}")
    print(f"📝 Prompts: {args.prompts}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at {args.model_path}")
        print("📝 Please make sure training has completed and model is saved.")
        return
    
    # Load the trained pipeline
    print("🔄 Loading trained Stable Diffusion pipeline...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None,  # Disable safety checker for Kanji generation
            requires_safety_checker=False
        )
        
        # Use CPU if CUDA not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        print(f"✅ Pipeline loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"🎲 Random seed set to {args.seed}")
    
    # Generate Kanji for each prompt
    print(f"\\n🎨 Generating Kanji for {len(args.prompts)} concepts...")
    
    all_results = []
    
    for i, prompt in enumerate(args.prompts):
        print(f"\\n📝 Generating Kanji for: '{prompt}'")
        
        prompt_results = []
        
        for sample_idx in range(args.num_samples):
            try:
                # Generate image
                with torch.autocast(device):
                    image = pipe(
                        prompt=prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        height=256,
                        width=256
                    ).images[0]
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"kanji_{prompt.replace(' ', '_')}_{sample_idx+1}_{timestamp}.png"
                filepath = os.path.join(args.output_dir, filename)
                image.save(filepath)
                
                prompt_results.append({
                    'prompt': prompt,
                    'sample': sample_idx + 1,
                    'filepath': filepath,
                    'success': True
                })
                
                print(f"  ✅ Sample {sample_idx+1} saved: {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to generate sample {sample_idx+1}: {e}")
                prompt_results.append({
                    'prompt': prompt,
                    'sample': sample_idx + 1,
                    'filepath': None,
                    'success': False,
                    'error': str(e)
                })
        
        all_results.extend(prompt_results)
        
        successful = sum(1 for r in prompt_results if r['success'])
        print(f"  📊 {successful}/{args.num_samples} samples generated successfully")
    
    # Summary
    total_successful = sum(1 for r in all_results if r['success'])
    total_attempted = len(all_results)
    
    print(f"\\n🎉 Generation Complete!")
    print(f"📊 Summary:")
    print(f"   • Total samples attempted: {total_attempted}")
    print(f"   • Successful generations: {total_successful}")
    print(f"   • Success rate: {total_successful/total_attempted*100:.1f}%")
    print(f"   • Output directory: {args.output_dir}")
    
    # Show some examples
    print(f"\\n🖼️  Generated Files:")
    successful_results = [r for r in all_results if r['success']]
    for result in successful_results[:10]:  # Show first 10
        print(f"   • {os.path.basename(result['filepath'])} - '{result['prompt']}'")
    
    if len(successful_results) > 10:
        print(f"   • ... and {len(successful_results) - 10} more files")
    
    print(f"\\n✨ Try opening the generated images to see your novel Kanji!")

if __name__ == "__main__":
    main()
