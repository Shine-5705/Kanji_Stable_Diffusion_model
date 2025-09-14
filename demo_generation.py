"""
Quick demo script to test Stable Diffusion pipeline with simple prompts
This uses the base pre-trained model to demonstrate the concept
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import time

def generate_sample_kanji():
    """Generate sample images using base Stable Diffusion to demonstrate concept"""
    
    print("🔄 Loading Stable Diffusion pipeline...")
    
    # Load the base pipeline (we'll use this to demonstrate the concept)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPMSolver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    print(f"✅ Using device: {device}")
    
    # Create output directory
    output_dir = "outputs/demo_generations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test prompts (these are to demonstrate the pipeline - actual Kanji would need fine-tuned model)
    demo_prompts = [
        "simple black calligraphy character on white background, minimal, clean lines",
        "traditional Japanese character, black ink on white paper, brush stroke style",
        "geometric symbol, black lines on white background, minimal design",
        "abstract character design, black strokes, white background",
        "simple logographic symbol, black on white, clean minimal design"
    ]
    
    print("🎨 Generating demo images...")
    
    generator = torch.Generator(device=device).manual_seed(42)
    
    for i, prompt in enumerate(demo_prompts):
        print(f"Generating image {i+1}/{len(demo_prompts)}: '{prompt[:50]}...'")
        
        start_time = time.time()
        
        with torch.autocast(device):
            images = pipe(
                prompt=prompt,
                num_inference_steps=20,  # Faster generation
                guidance_scale=7.5,
                num_images_per_prompt=2,
                generator=generator
            ).images
        
        generation_time = time.time() - start_time
        
        # Save images
        for j, image in enumerate(images):
            filename = f"demo_{i:02d}_{j:02d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  💾 Saved: {filename} ({generation_time:.1f}s)")
    
    print(f"\n✅ Demo generation complete! Images saved to: {output_dir}")
    print("\n📝 Note: These are demo images using the base model.")
    print("   For actual Kanji generation, the model needs to complete training.")
    
    return output_dir

if __name__ == "__main__":
    print("🚀 Kanji Stable Diffusion - Demo Generation")
    print("=" * 60)
    
    try:
        output_dir = generate_sample_kanji()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print(f"📁 Check the generated images in: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        print("💡 This might be due to memory constraints or missing dependencies.")
        print("   Try reducing batch size or using CPU mode.")
