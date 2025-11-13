# Updated Image Editing Pipeline Structure

## Overview
This pipeline performs controllable vehicle-focused editing with background regeneration through a 4-stage process, orchestrated either from the CLI (`python main.py`) or the FastAPI service (`api.py`).

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
│  • Model: UNet (`model/unet_coco_best.pth`)                     │
│  • Output: `stage1_mask.png` (stored at `images/` or `images/<run_id>/`) │
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
│  • Prompt: custom vehicle text prompt (UI/CLI configurable)     │
│  • Output: `stage2_vehicle.png` (edited vehicle)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Re-segmentation (UNet)                                │
│  ────────────────────────────────────────────────────────       │
│  • Function: segment_edited_image()                             │
│  • Purpose: Create fresh mask for newly generated vehicle       │
│  • Model: UNet (`model/unet_coco_best.pth`)                     │
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
│  • Prompt: custom background prompt (UI/CLI configurable)       │
│  • Output: `stage4_final.png` (FINAL IMAGE)                     │
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

> **Note:** The API stores each run in `images/<run_id>/stage*_*.png`, while the CLI writes to the shared `images/` folder.

## Usage

```bash
# Command-line pipeline
python main.py

# FastAPI service (serves the web UI at http://localhost:8000/frontend/index.html)
uvicorn api:app --reload
```

Ensure you have:
- Input image (e.g., `0a0e3fb8f782_01.jpg`) available when running the CLI
- UNet model: `model/unet_coco_best.pth`
- Stable Diffusion snapshot: `model/stable-diffusion/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/`

## Customization

Prompts are configurable from the CLI (`main.py`) and the web UI:

```python
# main.py defaults
vehicle_prompt = "black sedan car, photorealistic, high quality, detailed"
background_prompt = "beautiful mountain road background, golden hour, scenic ocean view, high quality, photorealistic"
```

**Negative prompts** are exposed through dropdown presets in the web UI (vehicle + background) and can be overridden via API parameters for fine-grained control.
