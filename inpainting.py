import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from PIL import Image


if __name__ == "__main__":
    # Downloading all the models may take a while
    # You can change the model to any model which supports inpainting
    # https://huggingface.co/models?sort=downloads&search=inpainting
    # Look at model description to see example of how to use it
    # Maybe you will need to change something (like scheduler)
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        pipeline = pipeline.to(device)

    # Use your own image
    image = Image.open("image.jpg") 
    # You can generate mask here 
    # https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint#create-a-mask-image
    mask = Image.open("mask.webp")

    result = pipeline(
        image=image,
        mask_image=mask,
        prompt="",
        inpainting_prompt="A young man with stylish black irokez haircut and high fade",
        num_inference_steps=30,  # play around with this parameter 
    )
    inpainted_image = result.images[0]
    resized_image = inpainted_image.resize(image.size)

    resized_image.save("inpainted.png")
