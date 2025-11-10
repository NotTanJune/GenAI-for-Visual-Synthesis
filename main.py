import torch
import numpy as np
from PIL import Image
import gc
import os
import shutil
from typing import Optional

# For Stable Diffusion inpainting
from diffusers import StableDiffusionInpaintPipeline

# Directory where pipeline images are stored
IMAGES_DIR = os.path.join(os.getcwd(), "images")


def prepare_images_dir(images_dir: Optional[str] = None):
    """Create images folder if missing. If it contains files, clear them.

    This ensures each run starts with an empty `images/` directory.
    """
    if images_dir is None:
        images_dir = IMAGES_DIR

    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
        print(f"Created images directory: {images_dir}")
        return

    # If folder exists and is not empty, clear its contents
    entries = os.listdir(images_dir)
    if entries:
        print(f"Images directory not empty ({len(entries)} items). Clearing...")
        for name in entries:
            path = os.path.join(images_dir, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
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
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output layer
        self.out = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        return torch.sigmoid(self.out(dec1))

def segment_image(img_path, model_path):
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
    mask_output = os.path.join(IMAGES_DIR, "stage1_mask.png")
    mask_pil.save(mask_output)

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    return mask_output

# --- Step 2: Vehicle Regeneration (Stable Diffusion Inpainting) ---
def regenerate_vehicle(img_path, mask_path, model_dir, prompt):
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

    # Load image and mask
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Use mask directly: white = areas to regenerate (vehicle), black = keep (background)
    # No inversion needed for vehicle regeneration

    # Generate new vehicle with custom prompt
    negative_prompt = "blurry, low quality, distorted, ugly, SUV, truck"
    
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=0.9  # Higher strength to fully replace the vehicle
    ).images[0]
    
    vehicle_output = os.path.join(IMAGES_DIR, "stage2_vehicle.png")
    result_img.save(vehicle_output)

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return result_img, vehicle_output

# --- Step 3: Re-segment the edited image ---
def segment_edited_image(img_path, model_path, output_name="stage3_mask.png"):
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
    if not os.path.isabs(output_name):
        output_name = os.path.join(IMAGES_DIR, output_name)
    mask_pil.save(output_name)

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ Re-segmentation mask saved to: {output_name}")
    return output_name

# --- Step 4: Background Inpainting (Stable Diffusion) ---
def inpaint_background(img_path, mask_path, model_dir):
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

    # Load image and mask
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Invert mask: white = areas to inpaint (background/black zone), black = keep (foreground/white zone)
    mask_array = np.array(mask)
    inverted_mask = 255 - mask_array
    mask_inverted = Image.fromarray(inverted_mask).convert("L")

    # Generate inpainted background
    prompt = "beautiful sunset beach background, golden hour, scenic ocean view, high quality, photorealistic"
    negative_prompt = "blurry, low quality, distorted, car, vehicle"
    
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_inverted,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    final_output = os.path.join(IMAGES_DIR, "stage4_final.png")
    result_img.save(final_output)

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"✓ Final image with inpainted background saved to: {final_output}")
    return result_img, final_output

# --- Entire Workflow ---
if __name__ == "__main__":
    img_path = "0a0e3fb8f782_01.jpg"
    model_folder = "model"
    
    # Path to UNet model for segmentation
    unet_path = os.path.join(model_folder, "unet_model.pth")
    
    # Path to Stable Diffusion model
    sd_model_path = os.path.join(
        model_folder, 
        "stable-diffusion",
        "models--runwayml--stable-diffusion-v1-5",
        "snapshots",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
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
    print(f"  1. Initial segmentation → {os.path.join(IMAGES_DIR, 'stage1_mask.png')}")
    print(f"  2. Vehicle editing → {os.path.join(IMAGES_DIR, 'stage2_vehicle.png')}")
    print(f"  3. Re-segmentation → {os.path.join(IMAGES_DIR, 'stage3_mask.png')}")
    print(f"  4. Background inpainting → {os.path.join(IMAGES_DIR, 'stage4_final.png')}")

