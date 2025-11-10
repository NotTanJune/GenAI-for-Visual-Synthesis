# GenAI for Visual Synthesis - AI-Powered Image Editing Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-1.5-orange.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Adigo10/GenAI--for--Visual--Synthesis-blue.svg)](https://github.com/Adigo10/GenAI-for-Visual-Synthesis)

An advanced AI-powered image editing pipeline that combines semantic segmentation with generative AI to transform images by changing both background and foreground elements independently. Perfect for creative content generation, product photography, and visual synthesis tasks.

## 🌟 Features

- **Semantic Segmentation**: U-Net based foreground-background separation
- **Background Inpainting**: Generate new backgrounds using Stable Diffusion
- **Object Regeneration**: Transform objects (vehicles, etc.) using AI
- **Intelligent Composition**: Seamlessly combine generated elements
- **GPU Acceleration**: Support for DirectML (AMD GPUs) and CUDA
- **Modular Pipeline**: Easy to customize and extend

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Adigo10/GenAI-for-Visual-Synthesis.git
   cd GenAI-for-Visual-Synthesis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models**
   ```bash
   python setup.py
   ```

### Usage

1. **Place your image** in the project root (e.g., `input_image.jpg`)

2. **Update the image path** in `main.py`:
   ```python
   img_path = "your_image.jpg"  # Change this to your image file
   ```

3. **Run the pipeline**
   ```bash
   python main.py
   ```

Note: The pipeline now writes all intermediate and final images into an `images/` folder created next to the working directory. On each run the script will create `images/` if missing and will clear any existing files inside it so every run starts with an empty `images/` folder.

## 📋 Pipeline Stages

### Stage 1: Initial Segmentation (UNet / SAM)
- **Model**: U-Net (or optionally SAM for segmentation assistance)
- **Input**: Original image
- **Output**: Binary mask (`images/stage1_mask.png`)
- **Purpose**: Identify the vehicle/foreground region in the original image (white = vehicle, black = background)

### Stage 2: Vehicle Regeneration (Stable Diffusion Inpainting)
- **Model**: Stable Diffusion Inpainting
- **Input**: Original image + Stage 1 mask
- **Output**: Edited vehicle image (`images/stage2_vehicle.png`)
- **Purpose**: Apply edits to the vehicle first (change model, color, style, etc.) using the mask to constrain edits to the vehicle area

### Stage 3: Re-segmentation (UNet)
- **Model**: U-Net
- **Input**: The edited vehicle image (Stage 2 output)
- **Output**: Fresh segmentation mask for the edited vehicle (`images/stage3_mask.png`)
- **Purpose**: Create an accurate mask that fits the newly generated vehicle so the background can be regenerated around it

### Stage 4: Background Inpainting (Stable Diffusion Inpainting)
- **Model**: Stable Diffusion Inpainting
- **Input**: Edited vehicle image (Stage 2) + Stage 3 mask (inverted to indicate background areas)
- **Output**: Final image with regenerated background (`images/stage4_final.png`)
- **Purpose**: Generate a new background that naturally fits the edited vehicle; this is done last so the background adapts to the final vehicle appearance

## 🎨 Example Transformations

| Original | Segmentation Mask | Edited Vehicle | Re-seg Mask | Final Result |
|----------|------------------|----------------|------------:|--------------|
| ![Original](0a0e3fb8f782_01.jpg) | ![Mask](images/stage1_mask.png) | ![Vehicle](images/stage2_vehicle.png) | ![Mask2](images/stage3_mask.png) | ![Final](images/stage4_final.png) |

**Example**: SUV → Black Sedan; pipeline edits vehicle first, re-segments it, then regenerates the background to match the edited vehicle.

## 🔧 Configuration

### Customizing Prompts

Edit the prompts in `main.py`:

```python
# Background generation prompt
background_prompt = "beautiful sunset beach background, golden hour, scenic ocean view"

# Object transformation prompt
vehicle_prompt = "luxury sedan car, photorealistic, high quality, professional photography"
```

### Model Paths

The pipeline uses local models stored in the `model/` directory:

```
model/
├── unet_model.pth                    # U-Net segmentation model
└── stable-diffusion/                 # Stable Diffusion models
    └── models--runwayml--stable-diffusion-v1-5/
        └── snapshots/
            └── 451f4fe16113bff5a5d2269ed5ad43b0592e9a14/
```

### Hardware Acceleration

The pipeline automatically detects and uses:
- **DirectML**: For AMD GPUs on Windows
- **CUDA**: For NVIDIA GPUs
- **CPU**: Fallback for systems without GPU

## 📁 Project Structure

```
GenAI-for-Visual-Synthesis/
├── main.py                          # Main pipeline script
├── setup.py                         # Model download script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── 0a0e3fb8f782_01.jpg              # Sample input image
├── model/                           # Pre-trained models
│   ├── unet_model.pth              # U-Net weights
│   └── stable-diffusion/           # Stable Diffusion models
├── archive/                         # Additional utilities
│   ├── extract_outdoor_data.py     # Data extraction script
│   └── genai-depth-estimation.ipynb # Depth estimation notebook
├── src/                            # Source code (if any)
├── tested_images/                  # Test results
└── stage[1-4]_*.png               # Pipeline outputs
```

## 🛠️ Technical Details

### U-Net Architecture
```python
# Encoder: 4 downsampling blocks (64 → 128 → 256 → 512)
# Bottleneck: 1024 channels
# Decoder: 4 upsampling blocks with skip connections
# Output: Single channel segmentation mask
```

### Stable Diffusion Configuration
- **Model**: `runwayml/stable-diffusion-v1-5`
- **Inference Steps**: 50
- **Guidance Scale**: 7.5
- **Strength**: 0.8-0.9 (for inpainting)

### Memory Management
- Automatic GPU memory cleanup after each stage
- Garbage collection between pipeline steps
- Model unloading when not in use

## 🎯 Use Cases

- **Automotive**: Change vehicle types (SUV → Sedan, Car → Truck)
- **Real Estate**: Transform property backgrounds
- **Fashion**: Update clothing/outfits on models
- **Product Photography**: Change product settings
- **Art**: Create surreal compositions
- **Content Creation**: Generate variations for marketing

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce inference steps or use CPU
num_inference_steps=30  # Instead of 50
```

**2. Model Download Issues**
```bash
# Manual download if setup.py fails
python setup.py
```

**3. Poor Segmentation Quality**
- Ensure input image is well-lit
- Try different threshold values in segmentation
- Consider retraining the U-Net model

**4. Inconsistent Results**
- Use higher guidance_scale (8.0-10.0)
- Experiment with different prompts
- Try multiple runs with same settings

### Performance Optimization

- **GPU**: Use CUDA/DirectML for 3-5x speedup
- **CPU**: Reduce inference steps to 30-40
- **Memory**: Process one image at a time
- **Quality**: Balance steps vs. quality based on hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Diffusion**: [Runway ML](https://runwayml.com/) for the base model
- **Diffusers**: [Hugging Face](https://huggingface.co/) for the inference library
- **PyTorch**: [Facebook AI](https://pytorch.org/) for the deep learning framework

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code comments in `main.py`

---

**Made with ❤️ using cutting-edge AI technology**