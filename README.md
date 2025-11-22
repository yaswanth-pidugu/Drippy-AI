# Drippy-AI

End-to-end outfit generation and product-search pipeline built with diffusion models, ControlNet, and an LLM reasoning layer.

## Purpose
Generates an image of an outfit based on a user prompt, extracts pose/control information, runs Stable Diffusion with ControlNet to synthesize final visuals, and uses an LLM to produce purchasable item lists with direct search links.

## Core Components
- **Stable Diffusion + ControlNet** for image generation using OpenPose conditioning.  
- **diffusers** for model loading, scheduling, and inference.  
- **controlnet_aux** for pose detection.  
- **LangChain + Google Gemini API** for item-list generation.  
- **Flipkart Search URL Builder** for fast product lookup.  
- **Utility stack:** NumPy, Pillow, OpenCV (cv2), torch.

## Pipeline Flow
1. Load input image or URL.  
2. Extract pose using `OpenposeDetector`.  
3. Build ControlNet model and Stable Diffusion pipeline.  
4. Generate controlled output using prompt + pose.  
5. Pass outfit description to LLM.  
6. Convert LLM items into encoded Flipkart search links.  
7. Return final outfit items + search URLs.

## Notebook
Open and run the project notebook at:
`/mnt/data/Drippy_AI_VIT_AP_CSI_Workshop.ipynb`

## Notebook Structure
- Environment setup and dependency installation.  
- Image loading and pose extraction.  
- ControlNet + Stable Diffusion pipeline initialization.  
- Inference functions for generating the final controlled image.  
- LLM prompt template, structured output parsing, and chain invocation.  
- Flipkart link generator for each predicted item.

## Requirements
- Python 3.10+  
- GPU with CUDA support (recommended)  
- Installable packages (example):  
  `diffusers`, `transformers`, `controlnet_aux`, `torch`, `opencv-python`, `Pillow`, `langchain`, `langchain-google-genai`, `numpy`, `accelerate`, `xformers`

## Setup
1. Create and activate a virtual environment.  
2. Install dependencies:
```bash
pip install -r requirements.txt
