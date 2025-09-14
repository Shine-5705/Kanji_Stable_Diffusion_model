"""
GPU-optimized rigorous training script for Kanji Stable Diffusion
High-performance training with proper GPU utilization
"""

import argparse
import logging
import math
import os
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from datasets import Dataset
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
import numpy as np

from dataset import KanjiDatasetHF

# Will error if the minimal version of diffusers is not installed
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-optimized Kanji Stable Diffusion training")
    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Kanji dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./models/kanji-sd-gpu", help="Directory to save the trained model")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for training images")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=2000, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--scale_lr", action="store_true", help="Scale learning rate by batch size and accumulation")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--prediction_type", type=str, default=None, help="Prediction type for scheduler")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--save_steps", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Checkpoint frequency")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Enable xformers memory efficient attention")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Validation
    if args.dataset_path is None:
        raise ValueError("--dataset_path is required")
    
    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }

def main():
    args = parse_args()
    
    # Check for GPU availability and adjust settings
    device_available = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adjust settings based on device
    if device_available == "cpu":
        print("⚠️  No GPU detected, adjusting settings for CPU training...")
        args.mixed_precision = "no"  # FP16 not stable on CPU
        args.train_batch_size = 1    # Reduce batch size for CPU
        args.dataloader_num_workers = 0  # No multiprocessing on CPU
        args.gradient_accumulation_steps = 1  # Simplify for CPU
        print(f"✅ Adjusted for CPU: batch_size=1, mixed_precision=no, workers=0")
    else:
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize accelerator with device-appropriate settings
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
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"🚀 Starting GPU-optimized Kanji Stable Diffusion training")
    logger.info(f"📊 Training configuration: {args}")
    logger.info(f"🔥 Using device: {accelerator.device}")
    logger.info(f"💾 Mixed precision: {args.mixed_precision}")
    
    # Load Stable Diffusion models
    logger.info("🔄 Loading Stable Diffusion models...")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    logger.info("✅ All models loaded successfully")
    
    # Freeze VAE and text encoder parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Enable memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("✅ Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"⚠️ Failed to enable xformers: {e}")
    
    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()
    
    # Set training mode
    unet.train()
    
    logger.info("🔒 VAE and text encoder frozen, UNet set to training mode")
    
    # Load and prepare dataset
    logger.info(f"📚 Loading Kanji dataset from {args.dataset_path}")
    
    dataset = KanjiDatasetHF(
        dataset_path=args.dataset_path,
        image_size=args.resolution,
        tokenizer=tokenizer
    )
    
    logger.info(f"✅ Dataset created with {len(dataset)} samples")
    
    # Create data loader with device-appropriate settings
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),  # Only pin memory if GPU available
    )
    
    # Scale learning rate by batch size and accumulation
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
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
    logger.info(f"   📊 Dataset size: {len(dataset)}")
    logger.info(f"   🔢 Batch size: {args.train_batch_size}")
    logger.info(f"   🔄 Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"   📏 Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    logger.info(f"   📈 Learning rate: {args.learning_rate}")
    logger.info(f"   🎯 Total training steps: {args.max_train_steps}")
    logger.info(f"   📚 Steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"   🔁 Number of epochs: {args.num_train_epochs}")
    
    # Prepare for distributed training
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move VAE and text encoder to accelerator device with appropriate dtype
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize the trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("kanji-stable-diffusion")
    
    # Start training
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("🎯 ***** Running training *****")
    logger.info(f"  📊 Num examples = {len(dataset)}")
    logger.info(f"  📚 Num Epochs = {args.num_train_epochs}")
    logger.info(f"  🔢 Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  📈 Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  🔄 Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  🎯 Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if available
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step - (first_epoch * len(train_dataloader))
            
            logger.info(f"📂 Resuming from checkpoint {path}")
            logger.info(f"🔄 Resuming at epoch {first_epoch}, step {global_step}")
    
    # Training loop
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    # Default to epsilon prediction
                    target = noise
                
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute MSE loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Gather the losses across all processes for logging (if we use distributed training)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # Log training metrics
                if global_step % 50 == 0:
                    logger.info(f"🎯 Step {global_step}/{args.max_train_steps}, Loss: {avg_loss.item():.6f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"💾 Saved checkpoint to {save_path}")
                
                # Save model periodically
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        unet_save = accelerator.unwrap_model(unet)
                        save_path = os.path.join(args.output_dir, f"unet-{global_step}")
                        unet_save.save_pretrained(save_path)
                        logger.info(f"💾 Saved UNet to {save_path}")
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        
        # Save final UNet
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        
        # Create and save final pipeline
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
