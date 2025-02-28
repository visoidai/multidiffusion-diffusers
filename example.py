from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    ModelMixin,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image
from io import BytesIO
from textEmbedder import text_embeddings
import torch
import os

from MultiDiffusionPipeline import MultiStableDiffusion

from mask_positions import get_mask_positions

# this was never tested lol


def get_device_and_dtype():
    device = "cpu"
    dtype = torch.float16
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # dtype = torch.float32
    else:
        raise ValueError("WARNING: need to run on GPU")
    return device, dtype


device, dtype = get_device_and_dtype()

pipe = MultiStableDiffusion.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    local_files_only=True,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# Use 4 prompts
prompt = ["a beautiful park", "a beautiful sky", "a bustling city", "a serene mountain"]
negative_prompt = [
    "an empty park",
    "a gloomy sky",
    "a deserted city",
    "a barren mountain",
]

# Generate prompt embeddings for each prompt
promptE = []
for i in range(len(prompt)):
    cond, _ = text_embeddings(pipe, prompt[i], "", clip_stop_at_last_layers=2)
    promptE.append(cond)

# Define 4 mask positions.
# (Format: "x0:y0-x1:y1" in pixels; adjust these coordinates as needed for your image)
# pos = [
#     "0:0-256:256",  # Top-left quarter
#     "256:0-512:256",  # Top-right quarter
#     "0:256-256:512",  # Bottom-left quarter
#     "256:256-512:512",  # Bottom-right quarter
# ]
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "Untitled.png")
mask_positions = get_mask_positions(filename)
pos = list(mask_positions.values())
print(list(mask_positions.values()))

# Define 4 mask types (these serve as strengths, similar to z-index values)
mask_types = [1, 12, 5, 8]

# Load an image to be used as one of the mask inputs (for the one non-rectangular mask if needed)

with open(filename, "rb") as buffer:
    buffer.seek(0)
    image_bytes = buffer.read()
imageM = Image.open(BytesIO(image_bytes))

# If you prefer a PIL image for a mask position, you can place it in the pos list.
# Here we use four positions defined as strings.
# Alternatively, you can mix strings and images as needed.
# Example: pos = ["0:0-256:256", imageM, "0:256-256:512", "256:256-512:512"]

# Set a manual seed for reproducibility.
generator = torch.manual_seed(2733424006)

# Call the pipeline with 4 prompts and their masking positions.
output = pipe(
    prompt=None,  # we are using prompt_embeds directly
    negative_prompt=None,
    prompt_embeds=promptE,
    pos=pos,
    mask_types=mask_types,
    height=512,
    width=512,
    num_inference_steps=20,
    generator=generator,
    guidance_scale=7.0,
)
# Get the generated image (first in the batch)
image = output.images[0]

# Save the generated image.
image.save("output.png")
