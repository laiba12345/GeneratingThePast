import imageio.v3 as iio
from PIL import Image
import numpy as np
from diffusers.utils import export_to_video
import os
import torch
from diffusers import LTXImageToVideoPipeline

video_path = "/Data/Laiba-Maddy-Project/Project2/Dataset/dataset2/videos/15081.webm"
video = iio.imread(video_path)

BASE_MODEL = "/Data/Laiba-Maddy-Project/Project2/models/LTX-Video"
LORA_PATH = "/Data/Laiba-Maddy-Project/Project2/outputs/output2/lora_weights/001500/pytorch_lora_weights.safetensors"  

# Optional we did this since our env was not working
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/ffmpeg-8.0/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

pipe = LTXImageToVideoPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load LoRA
pipe.load_lora_weights(LORA_PATH, adapter_name="reverse-lora")
pipe.set_adapters(["reverse-lora"], [1.4])
# first frame of reversed video = last frame of original video
conditioning_image = Image.fromarray(np.asarray(video[-1])).convert("RGB")

prompt="<REV> a bottle falling"
result = pipe(
        prompt=prompt,
      
        image=conditioning_image,
        width=320,
        height=224,
        num_frames=49,
        num_inference_steps=25,
        guidance_scale=3.2,
    )
frames = result.frames[0]
out_path = f"test_outputs/bottle falling like a rock.mp4"
export_to_video(frames, out_path, 
                fps=8)
print(f"saved {out_path} :: {prompt}")