"""
Inference script for generating novel Kanji from English definitions
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import argparse
import os
import random

def generate_kanji(
    model_path,
    prompts,
    output_dir="outputs/generated",
    num_inference_steps=50,
    guidance_scale=7.5,
    num_images_per_prompt=4,
    seed=None
):
    """Generate Kanji images from text prompts"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the fine-tuned pipeline
    print(f"Loading model from {model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPMSolver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None
    
    # Generate images for each prompt
    for i, prompt in enumerate(prompts):
        print(f"Generating Kanji for: '{prompt}'")
        
        # Generate images
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
        
        # Save images
        for j, image in enumerate(images):
            filename = f"kanji_{i:03d}_{j:02d}_{prompt.replace(' ', '_').replace(',', '')[:50]}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate novel Kanji from English definitions")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/kanji-sd",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "love",
            "peace",
            "internet",
            "artificial intelligence",
            "smartphone",
            "social media",
            "climate change",
            "renewable energy",
            "space exploration",
            "virtual reality",
            "machine learning",
            "cryptocurrency",
            "electric car",
            "youtube",
            "elon musk",
            "baby robot",
            "armed fish",
            "language model"
        ],
        help="Text prompts to generate Kanji for"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated",
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
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )
    
    args = parser.parse_args()
    
    # Generate Kanji
    generate_kanji(
        model_path=args.model_path,
        prompts=args.prompts,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed
    )
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
