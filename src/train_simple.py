"""
Simplified fine-tuning script for Stable Diffusion on Kanji generation
Compatible with Windows and CPU-only environments
"""

import argparse
import logging
import math
import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from datasets import Dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms

from dataset import KanjiDatasetHF

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on Kanji dataset")
    
    # Dataset args
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Kanji dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kanji-sd-model",
        help="Directory to save the trained model"
    )
    
    # Training args
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution for training images"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.dataset_path is None:
        raise ValueError("--dataset_path is required")
    
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Initialize accelerator (simplified for Windows compatibility)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=args.logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.logging_dir else None,
        project_config=accelerator_project_config,
    )
    
    logger.info(f"🚀 Starting Kanji Stable Diffusion training")
    logger.info(f"📊 Training configuration: {args}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    # Load the dataset
    logger.info(f"📚 Loading Kanji dataset from {args.dataset_path}")
    
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
        
        logger.info(f"📊 Dataset loaded: {len(dataset_data)} samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return
    
    # Load Stable Diffusion models first (needed for tokenizer)
    logger.info("🔄 Loading Stable Diffusion models...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        # Load tokenizer and text encoder
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        
        logger.info("✅ Tokenizer and text encoder loaded")
        
        # Create PyTorch dataset
        dataset = KanjiDatasetHF(
            dataset_path=args.dataset_path,
            image_size=args.resolution,
            tokenizer=tokenizer
        )
        
        logger.info(f"✅ Dataset created with {len(dataset)} samples")
        
        # Create data loader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        
        # Load scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        logger.info("✅ All models loaded successfully")
    
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Only train the UNet
    unet.train()
    
    logger.info("🔒 VAE and text encoder frozen, UNet set to training mode")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    logger.info(f"📈 Training setup:")
    logger.info(f"   Total training steps: {args.max_train_steps}")
    logger.info(f"   Steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Prepare for distributed training
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move VAE and text encoder to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training loop
    logger.info("🎯 Starting training loop...")
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"📚 Epoch {epoch + 1}/{args.num_train_epochs}")
        
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Get batch data
                pixel_values = batch["pixel_values"].to(accelerator.device)
                input_ids = batch["input_ids"].to(accelerator.device)
                
                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise to add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log progress
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}/{args.max_train_steps}, Loss: {loss.item():.6f}")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    logger.info(f"💾 Saving checkpoint at step {global_step}")
                    
                    # Create checkpoint directory
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save UNet
                    accelerator.unwrap_model(unet).save_pretrained(
                        os.path.join(checkpoint_dir, "unet")
                    )
                    
                    logger.info(f"✅ Checkpoint saved to {checkpoint_dir}")
                
                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    logger.info("💾 Saving final model...")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Save the UNet
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        
        # Create a complete pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        )
        
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"✅ Final model saved to {args.output_dir}")
    
    accelerator.end_training()
    logger.info("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
