from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import gc
from safetensors.torch import load_file
from google.colab import drive
from huggingface_hub import login

drive.mount('/content/drive')


model_name = "SG161222/Realistic_Vision_V5.1_noVAE"

if(torch.cuda.is_available()):

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        safety_checker = None
    ).to("cuda")



    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", final_sigmas_type="sigma_min")


    prompt = (
         "1 girl,  Sexy 18 year old girl ,  braids,  beautiful face, nice,  beautiful,  beautiful,  beautiful face,hair band"
         "blue-white skin, parts, Blonde, blue eyes, Perfect, glowing lips ,  Big Lips,  Thick lips,   half-open lips  ,uncensored,fullbody,"
         "Light Pink Lip Gloss  ,  skinny thighs, petite body, ,  skinny,  flat chest,  skinny,  body(body heigth:), Big Head,"
         "Innocent Looks  , Young Face, long pink nails,  choker, Legs and High Heels HD ,  Micro Plaid Skirt  ,   white tank top with shoulders out ,"
         "White Tight Gather Socks ,  BLACK PATENT LEATHER HIGH HEELS ,  sweaty ,  On your knees　In the bedroom　 Candles and Soft Warm Light . ( look up at the viewer ), "
         " The color of the room is pink and white ,   perfect eyes ,  very  detailed eyes,   beautiful and expressive  ,   detailed eyes , 35mm Photography, movie, Bokeh,"
         " professional, 4K,  high definition dynamic lighting,  photorealistic, 8k, born, Rico,    Intricate Details , naked,  topless"

    )


    negative_prompt = (
        "blurry, low quality, bad anatomy, deformed, unrealistic body proportions, multiple models, duplicated subject, twin, clone, extra face,extra hair,extra body part,"
        "extra limbs, extra fingers, fused fingers, mutated hands, bad hands, broken joints, twisted body, misaligned limbs, distorted fingers, "
        "extra clothes, misplaced clothing, misaligned clothing, unnatural clothing, unaligned body, missing body parts, weird posture,extra body structure "
        "unnatural skin texture, overexposed, underexposed, noisy, low resolution, artifacts, watermark, logo, text, signature, "
        "cartoon, CGI, 3D render, painting, drawing, sketch, unrealistic, poorly drawn, bad shading, shadows in wrong place"
    )


    # Face Portrait: 896x896
    # Portrait: 896x896, 768x1024
    # Half Body: 768x1024, 640x1152
    # Full Body: 896x896, 768x1024, 640x1152, 1024x768, 1152x640

    pipe.enable_attention_slicing()
    for i in range(4):
      image = pipe(
          prompt,
          negative_prompt=negative_prompt,
          guidance_scale=8,
          width=640,
          height=1152,
          num_inference_steps= 40,
          seed = 42,
      ).images[0]
      image.save(f"/content/drive/MyDrive/output{i+1}.png")



else:
  print('GPU is not available')








