from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import random

import os
import gc
from safetensors.torch import load_file
from google.colab import drive
from huggingface_hub import login

drive.mount('/content/drive')


# !nvidia-smi

def clean_up_gpu(variables=["pipe", "image", "model", "output"]):
    for name in variables:
        if name in globals():
            # Delete the variable from the global scope
            var = globals()[name]
            del globals()[name]

            # If the variable is a tensor, delete it from GPU memory
            if torch.is_tensor(var) and var.is_cuda:
                var.cpu()  # Move to CPU first to avoid potential errors
                del var

    # Run garbage collection and empty the CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
clean_up_gpu()

#login token 
# login(token="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")

model_name = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

if(torch.cuda.is_available()):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        safety_checker = None
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="sigma_min")

    prompt = (
      "Ultra-realistic, luxurious modern interior design of a spacious living room, flooded with natural light, featuring floor-to-ceiling windows, soft neutral tones with gold and black accents, plush velvet sofas, marble coffee table, indoor plants, elegant lighting fixtures, minimal yet warm décor, perfectly staged for a high-end lifestyle magazine, cinematic lighting, 8K resolution"
    )

    negative_prompt = (
       "realistic, photo, 3D, sketch, blurry, grainy, lowres, pixelated, low detail, bad anatomy, disfigured, ugly, extra limbs, distorted hands, watermark, text, signature, duplicate, dark, low contrast, weird colors, realism."
    )

    pipe.enable_attention_slicing()
    for i in range(4):
      image = pipe(
          prompt,
          negative_prompt=negative_prompt,
          guidance_scale=8,
          width=768,
          height=1024,
          num_inference_steps= 40,
          generator=torch.manual_seed(random.randint(0, 99999))
      ).images[0]
      image.save(f"/content/drive/MyDrive/output{i+1}.png")



else:
  print('GPU is not available')


# Face Portrait: 896x896
# Portrait: 896x896, 768x1024
# Half Body: 768x1024, 640x1152
# Full Body: 896x896, 768x1024, 640x1152, 1024x768, 1152x640






