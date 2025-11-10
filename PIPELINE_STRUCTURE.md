# Updated Image Editing Pipeline Structure

## Overview
This pipeline performs sophisticated vehicle editing with background regeneration through a 4-stage process.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Original Image                        │
│                    (0a0e3fb8f782_01.jpg)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Initial Segmentation (UNet)                           │
│  ────────────────────────────────────────────────────────       │
│  • Function: segment_image()                                    │
│  • Purpose: Identify vehicle region in original image           │
│  • Model: UNet (unet_model.pth)                                 │
│  • Output: stage1_mask.png                                      │
│  • Mask: White = vehicle, Black = background                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Vehicle Regeneration (Stable Diffusion)               │
│  ────────────────────────────────────────────────────────       │
│  • Function: regenerate_vehicle()                               │
│  • Purpose: Generate edits for car/vehicle                      │
│  • Model: Stable Diffusion Inpainting                           │
│  • Input: Original image + Stage 1 mask                         │
│  • Prompt: "black sedan car, photorealistic, high quality..."  │
│  • Output: stage2_vehicle.png (edited vehicle)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Re-segmentation (UNet)                                │
│  ────────────────────────────────────────────────────────       │
│  • Function: segment_edited_image()                             │
│  • Purpose: Create fresh mask for newly generated vehicle       │
│  • Model: UNet (unet_model.pth)                                 │
│  • Input: Stage 2 vehicle image                                 │
│  • Output: stage3_mask.png                                      │
│  • Mask: White = new vehicle, Black = background                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Background Inpainting (Stable Diffusion)              │
│  ────────────────────────────────────────────────────────       │
│  • Function: inpaint_background()                               │
│  • Purpose: Generate new background around edited vehicle       │
│  • Model: Stable Diffusion Inpainting                           │
│  • Input: Stage 2 vehicle + Stage 3 mask (inverted)             │
│  • Prompt: "beautiful sunset beach background, golden hour..."  │
│  • Output: stage4_final.png (FINAL IMAGE)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  FINAL OUTPUT   │
                   │ stage4_final.png│
                   └─────────────────┘
```

## Key Changes from Previous Version

### Old Pipeline:
1. Segment original image
2. Inpaint background first
3. Regenerate vehicle
4. Combine results manually

### New Pipeline:
1. **Segment original image** (UNet/SAM)
2. **Regenerate vehicle first** (Stable Diffusion) - Vehicle edits happen early
3. **Re-segment the edited image** (UNet) - Fresh mask for new vehicle
4. **Inpaint background last** (Stable Diffusion) - Background adapts to vehicle

## Advantages of New Structure

1. **Vehicle-First Approach**: Vehicle editing happens before background generation, allowing the background to adapt to the new vehicle
2. **Accurate Masking**: Re-segmentation ensures the mask fits the newly generated vehicle perfectly
3. **Better Integration**: Background inpainting can naturally blend around the edited vehicle
4. **Cleaner Results**: No manual combining needed - Stable Diffusion handles the final composition

## Output Files

| Stage | File Name | Description |
|-------|-----------|-------------|
| 1 | `stage1_mask.png` | Initial segmentation mask of original vehicle |
| 2 | `stage2_vehicle.png` | Edited/regenerated vehicle image |
| 3 | `stage3_mask.png` | Re-segmented mask of edited vehicle |
| 4 | `stage4_final.png` | **Final output** with edited vehicle + new background |

## Usage

```bash
python main.py
```

Ensure you have:
- Input image: `0a0e3fb8f782_01.jpg`
- UNet model: `model/unet_model.pth`
- Stable Diffusion model in: `model/stable-diffusion/...`

## Customization

You can modify the prompts in `main.py`:

- **Vehicle prompt** (line 268): Change vehicle appearance
- **Background prompt** (line 145): Change background scene
- **Negative prompts**: Control what to avoid in generation
