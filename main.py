import torch
import numpy as np
from PIL import Image
import gc
import os
import shutil
from pathlib import Path
from typing import Optional

# For Stable Diffusion inpainting
from diffusers import StableDiffusionInpaintPipeline

# Directory where pipeline images are stored
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"


def prepare_images_dir(images_dir: Optional[Path] = None):
    """Create images folder if missing. If it contains files, clear them.

    This ensures each run starts with an empty `images/` directory.
    """
    if images_dir is None:
        images_dir = IMAGES_DIR

    images_dir = Path(images_dir)

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created images directory: {images_dir}")
        return

    # If folder exists and is not empty, clear its contents
    entries = list(images_dir.iterdir())
    if entries:
        print(f"Images directory not empty ({len(entries)} items). Clearing...")
        for path in entries:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: failed to remove {path}: {e}")
        print("Images directory cleared")
    else:
        print("Images directory exists and is empty")

# --- Step 1: Foreground-Background Segmentation (U-Net) ---
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = torch.nn.Conv2d(64, out_channels, 1)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output with sigmoid activation
        out = torch.sigmoid(self.out(d1))
        return out

def segment_image(img_path, model_path, output_dir: Optional[Path] = None, output_name: str = "stage1_mask.png"):
    # Load model
    unet = UNet()
    unet.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    unet.eval()

    image = Image.open(img_path).convert("RGB").resize((256, 256))
    img_tensor = torch.tensor(
        (np.array(image) / 255.0).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        mask = unet(img_tensor).squeeze().numpy()
    mask_img = (mask > 0.5).astype("uint8") * 255
    mask_pil = Image.fromarray(mask_img).convert("L")
    # Save into images/ folder
    output_dir = Path(output_dir) if output_dir else IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output = output_dir / output_name
    mask_pil.save(mask_output)

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    return str(mask_output)

# --- Step 2: Vehicle Regeneration (Stable Diffusion Inpainting) ---
def regenerate_vehicle(img_path, mask_path, model_dir, prompt, negative_prompt="blurry, low quality, distorted, ugly car, deformed vehicle", output_dir: Optional[Path] = None, output_name: str = "stage2_vehicle.png"):
    """
    Use Stable Diffusion to regenerate the vehicle (white zone of mask).
    White in mask = vehicle to regenerate
    Black in mask = background to keep
    """
    # Load Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # Use DirectML for AMD GPUs on Windows, or CPU
    try:
        import torch_directml
        device = torch_directml.device()
        pipe = pipe.to(device)
        print("Using DirectML (AMD GPU)")
    except:
        device = "cpu"
        pipe = pipe.to(device)
        print("Using CPU")

    # Load image and mask, resize to 256x256
    image = Image.open(img_path).convert("RGB").resize((256, 256))
    mask = Image.open(mask_path).convert("L").resize((256, 256))
    
    # Use mask directly: white = areas to regenerate (vehicle), black = keep (background)
    # No inversion needed for vehicle regeneration

    # Generate new vehicle with custom prompt
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        mask_threshold=0.3,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        guidance_scale=10,
        # strength=0.9  # Higher strength to fully replace the vehicle
    ).images[0]
    
    output_dir = Path(output_dir) if output_dir else IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    vehicle_output = output_dir / output_name
    result_img.save(vehicle_output)

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return result_img, str(vehicle_output)

# --- Step 3: Re-segment the edited image ---
def segment_edited_image(img_path, model_path, output_dir: Optional[Path] = None, output_name="stage3_mask.png"):
    """
    Perform segmentation again on the edited vehicle image.
    This creates a fresh mask for the newly generated vehicle.
    """
    # Load model
    unet = UNet()
    unet.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    unet.eval()

    image = Image.open(img_path).convert("RGB").resize((256, 256))
    img_tensor = torch.tensor(
        (np.array(image) / 255.0).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        mask = unet(img_tensor).squeeze().numpy()
    mask_img = (mask > 0.5).astype("uint8") * 255
    mask_pil = Image.fromarray(mask_img).convert("L")
    # Ensure saved to images folder
    output_dir = Path(output_dir) if output_dir else IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    mask_pil.save(output_path)

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ Re-segmentation mask saved to: {output_path}")
    return str(output_path)

# --- Step 4: Background Inpainting (Stable Diffusion) ---
def inpaint_background(img_path, mask_path, model_dir, prompt="beautiful mountain road background, golden hour, scenic ocean view, high quality, photorealistic", negative_prompt="blurry, low quality, distorted, car, vehicle, text, watermark", output_dir: Optional[Path] = None, output_name: str = "stage4_final.png"):
    """
    Use Stable Diffusion inpainting to fill in the background (black zone of mask).
    Black in mask = background to inpaint
    White in mask = foreground to keep
    """
    # Load Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # Use DirectML for AMD GPUs on Windows, or CPU
    try:
        import torch_directml
        device = torch_directml.device()
        pipe = pipe.to(device)
        print("Using DirectML (AMD GPU)")
    except:
        device = "cpu"
        pipe = pipe.to(device)
        print("Using CPU")

    # Load image and mask, resize to 256x256
    image = Image.open(img_path).convert("RGB").resize((256, 256))
    mask = Image.open(mask_path).convert("L").resize((256, 256))
    
    # Invert mask: white = areas to inpaint (background/black zone), black = keep (foreground/white zone)
    mask_array = np.array(mask)
    inverted_mask = 255 - mask_array
    mask_inverted = Image.fromarray(inverted_mask).convert("L")

    # Generate inpainted background
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_inverted,
        num_inference_steps=50,
        guidance_scale=10
    ).images[0]
    
    output_dir = Path(output_dir) if output_dir else IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / output_name
    result_img.save(final_output)

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ Final image with inpainted background saved to: {final_output}")
    return result_img, str(final_output)

# --- Entire Workflow ---
if __name__ == "__main__":
    img_path = "0a2bbd5330a2_03.jpg"
    model_folder = BASE_DIR / "model"
    
    # Path to UNet model for segmentation
    unet_path = model_folder / "unet_coco_best.pth"
    
    # Path to Stable Diffusion model
    sd_model_path = (
        model_folder
        / "stable-diffusion"
        / "models--runwayml--stable-diffusion-v1-5"
        / "snapshots"
        / "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )

    print("Starting image editing pipeline...")
    print("="*60)
    # Ensure images directory exists and is cleared if needed
    prepare_images_dir(IMAGES_DIR)
    
    # Stage 1: Initial Segmentation (UNet/SAM)
    print("\n[Stage 1] Initial segmentation with UNet...")
    print("Purpose: Identify vehicle region in original image")
    mask_path = segment_image(img_path, unet_path)
    print(f"✓ Initial mask saved to: {mask_path}")

    # Stage 2: Vehicle Regeneration/Editing
    print("\n[Stage 2] Regenerating vehicle with Stable Diffusion...")
    print("Purpose: Generate edits for the car/vehicle")
    vehicle_prompt = "black sedan car, photorealistic, high quality, detailed"
    vehicle_img, vehicle_path = regenerate_vehicle(img_path, mask_path, sd_model_path, prompt=vehicle_prompt)
    print(f"✓ Edited vehicle image saved to: {vehicle_path}")

    # Stage 3: Re-segmentation on edited image
    print("\n[Stage 3] Re-segmenting the edited vehicle image with UNet...")
    print("Purpose: Create fresh mask for the newly generated vehicle")
    new_mask_path = segment_edited_image(vehicle_path, unet_path, output_name="stage3_mask.png")

    # Stage 4: Final Background Inpainting
    print("\n[Stage 4] Inpainting background with Stable Diffusion...")
    print("Purpose: Generate new background around the edited vehicle")
    final_img, final_path = inpaint_background(vehicle_path, new_mask_path, sd_model_path)

    print("\n" + "="*60)
    print(f"🎉 Pipeline complete! Final image: {final_path}")
    print("\nPipeline summary:")
    print(f"  1. Initial segmentation → {IMAGES_DIR / 'stage1_mask.png'}")
    print(f"  2. Vehicle editing → {IMAGES_DIR / 'stage2_vehicle.png'}")
    print(f"  3. Re-segmentation → {IMAGES_DIR / 'stage3_mask.png'}")
    print(f"  4. Background inpainting → {IMAGES_DIR / 'stage4_final.png'}")

