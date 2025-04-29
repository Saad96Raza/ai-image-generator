from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler 
import torch
import os
from safetensors.torch import load_file


model_name = "SG161222/Realistic_Vision_V6.0_B1_noVAE"


pipe = StableDiffusionPipeline.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    safety_checker=None
    ).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  

lora_path = '/content/drive/MyDrive/loras/photorealistic_facial_details_v1.safetensors'

if os.path.exists(lora_path):
    lora_state_dict = load_file(lora_path, device="cuda")
    
    for layer in pipe.unet.attn_processors:
        pipe.unet.attn_processors[layer] = lora_state_dict

    pipe.fuse_lora(lora_scale=0.7)
    print("LoRA loaded and fused successfully.")
else:
    print(f"LoRA file not found at {lora_path}")

pipe.num_inference_steps = 40  

prompt = (
    "masterpiece, best quality, 8k, RAW photo, DSLR, ultra-realistic portrait of a stunning young brunette woman outdoor,sharp detailed eyes"
    "wearing a red skirt and topless, natural soft lighting, smooth skin with fine pores,well-fitted clothes, properly aligned clothing, realistic proportions"
    "subtle expression, soft shadows, ambient cinematic light, depth of field, single model, only one woman, no duplicates"
    "bokeh, symmetrical face, sharp eyes, detailed hair strands, "
    "shot with Canon EOS 5D Mark IV, 85mm f/1.4 lens"
)
negative_prompt = (
    "blurry, low quality, deformed, bad anatomy, unrealistic body proportions, multiple model"
    "extra limbs, extra fingers, fused fingers, mutated hands, bad hands, twisted body"
    "extra clothes, misplaced clothes, unnatural clothing, unnatural skin texture, "
    "mutated, noisy, overexposed, underexposed, cartoon, CGI, painting, sketch, "
    "unrealistic, poorly drawn, distorted body, watermark, text, logo"
)


image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    guidance_scale=7,
    width=1800 ,     # <-- add width
    height=1800       # <-- add height
).images[0]  # Set guidance_scale here

image.show()

image.save("/content/drive/MyDrive/output.png")
