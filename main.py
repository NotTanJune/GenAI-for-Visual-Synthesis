import torch
import numpy as np
from PIL import Image
import gc
import os

# For Stable Diffusion inpainting
from diffusers import StableDiffusionInpaintPipeline

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
    mask_pil.save("stage1_mask.png")

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    return "stage1_mask.png"

# --- Step 2: Background Inpainting (Stable Diffusion) ---
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
    
    result_img.save("stage2_background.png")

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return result_img, "stage2_background.png"

# --- Step 3: Vehicle Regeneration (Stable Diffusion Inpainting) ---
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
    
    result_img.save("stage3_vehicle.png")

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return result_img, "stage3_vehicle.png"

# --- Step 4: Combine Background and Vehicle ---
def combine_results(background_img, vehicle_img, mask_path, output_path="final_combined.png"):
    """
    Combine the background from stage 2 and vehicle from stage 3.
    White in mask = use vehicle image
    Black in mask = use background image
    """
    # Load images
    if isinstance(background_img, str):
        background = Image.open(background_img).convert("RGB")
    else:
        background = background_img.convert("RGB")
    
    if isinstance(vehicle_img, str):
        vehicle = Image.open(vehicle_img).convert("RGB")
    else:
        vehicle = vehicle_img.convert("RGB")
    
    mask = Image.open(mask_path).convert("L")
    
    # Ensure all images are the same size
    size = background.size
    vehicle = vehicle.resize(size)
    mask = mask.resize(size)
    
    # Convert to numpy arrays
    bg_array = np.array(background).astype(np.float32)
    veh_array = np.array(vehicle).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3d = np.stack([mask_array, mask_array, mask_array], axis=2)
    
    # Combine: white mask areas = vehicle, black mask areas = background
    combined = (veh_array * mask_3d + bg_array * (1 - mask_3d)).astype(np.uint8)
    
    # Convert back to PIL image
    result = Image.fromarray(combined)
    result.save(output_path)
    
    print(f"✓ Combined image saved to: {output_path}")
    return output_path

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
    
    # Stage 1: Segmentation
    print("\n[Stage 1] Segmenting foreground/background...")
    mask_path = segment_image(img_path, unet_path)
    print(f"✓ Mask saved to: {mask_path}")

    # Stage 2: Inpaint Background (black zone of mask)
    print("\n[Stage 2] Inpainting background with Stable Diffusion...")
    background_img, background_path = inpaint_background(img_path, mask_path, sd_model_path)
    print(f"✓ Background image saved to: {background_path}")

    # Stage 3: Regenerate Vehicle (white zone of mask)
    print("\n[Stage 3] Regenerating vehicle with Stable Diffusion...")
    vehicle_prompt = "luxury sedan car, photorealistic, high quality"
    vehicle_img, vehicle_path = regenerate_vehicle(img_path, mask_path, sd_model_path, prompt=vehicle_prompt)
    print(f"✓ Vehicle image saved to: {vehicle_path}")

    # Stage 4: Combine Background and Vehicle
    print("\n[Stage 4] Combining background and vehicle...")
    final_path = combine_results(background_img, vehicle_img, mask_path, output_path="final_combined.png")

    print(f"\n🎉 Pipeline complete! Final image: {final_path}")
