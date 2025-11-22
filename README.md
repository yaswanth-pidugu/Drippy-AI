Drippy-AI
End-to-end outfit generation and product-search pipeline built with diffusion models, ControlNet, and an LLM reasoning layer.
Purpose
Generates an image of an outfit based on a user prompt, extracts pose/control information, runs Stable Diffusion with ControlNet to synthesize final visuals, and uses an LLM to produce purchasable item lists with direct search links.
Core Components


Stable Diffusion + ControlNet for image generation using OpenPose conditioning.


Diffusers for model loading, scheduling, and inference.


controlnet_aux for pose detection.


LangChain + Google Gemini API for item-list generation.


Flipkart Search URL Builder for fast product lookup.


Utility stack: NumPy, PIL, cv2, torch.


Pipeline Flow


Load input image or URL.


Extract pose using OpenposeDetector.


Build ControlNet model and SD pipeline.


Generate controlled output using prompt + pose.


Pass outfit description to LLM.


Convert LLM items into encoded Flipkart search links.


Return final outfit items + search URLs.


Notebook Structure


Environment setup and dependency installation.


Image loading and pose extraction.


ControlNet + SD pipeline initialization.


Inference functions for generating the final controlled image.


LLM prompt template, structured output parsing, and chain invocation.


Flipkart link generator for each predicted item.


Requirements


Python 3.10+


GPU with CUDA support


Packages installed via pip:
diffusers, transformers, controlnet_aux, torch, opencv-python, Pillow,
langchain, langchain-google-genai, numpy, accelerate, xformers.


Setup
Activate env and install dependencies:
pip install -r requirements.txt

Store Google Gemini key as:
export GOOGLE_API_KEY="your-key"

Run
Open the notebook:
jupyter notebook Drippy_AI_VIT_AP_CSI_Workshop.ipynb

Execute sequentially from top to bottom.
Output


AI-generated outfit image conditioned on provided pose.


Deterministic structured JSON list of outfit items from LLM.


Direct clickable Flipkart search links for each item.


Folder Expectations


Add sample input images under inputs/ if desired.


Generated outputs saved under outputs/ if paths are modified in notebook.


Notes


Model weights must be downloaded automatically during runtime.


Ensure GPU memory is sufficient for ControlNet (8–12 GB recommended).

