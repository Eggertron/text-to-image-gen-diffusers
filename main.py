# Import libraries
import argparse
import torch
from diffusers import DiffusionPipeline

parser = argparse.ArgumentParser(prog='Simple text to image diffuser')
parser.add_argument('prompt')
args = parser.parse_args()

# Set model checkpoint path
model_path = "model.ckpt"

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained(model_path)

# Define your text prompt
text_prompt = args.prompt

# Generate the image
image = pipe.text_to_image(text_prompt, guidance_scale=7.5)

# Save or display the image
image.save("output.png")
# or 
#image.show()
