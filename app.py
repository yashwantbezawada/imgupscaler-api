import io
import torch
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from models.swinir_network import SwinIR

###############################################################################
# SPEED / SAFETY TWEAKS
###############################################################################
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# 1) CREATE L x4 MODEL
###############################################################################
def create_swinir_l_x4_real() -> SwinIR:
    """
    'L' real-SwinIR for 4×:
      - embed_dim=240
      - depths=[6, 6, 6, 6, 6, 6, 6, 6, 6]
      - upsampler='nearest+conv'
      - resi_connection='3conv'
      - scale=4
    """
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2.0,
        upsampler='nearest+conv',
        resi_connection='3conv',
        in_chans=3,
        patch_size=1
    )
    return model


###############################################################################
# 2) SAFE LOAD FUNCTION
###############################################################################
def safe_load(model: SwinIR, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"Loading from {checkpoint_path}...")
    for key in ["params", "params_ema", "state_dict", "model"]:
        if key in ckpt:
            try:
                model.load_state_dict(ckpt[key], strict=False)
                print("Model loaded successfully.")
                return
            except RuntimeError as e:
                print(f"Error loading model: {e}")
    model.load_state_dict(ckpt, strict=False)


###############################################################################
# 3) CHECKPOINT PATHS
###############################################################################
PSNR_L_X4_PATH = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
GAN_L_X4_PATH = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"


###############################################################################
# 4) CREATE & LOAD MODELS
###############################################################################
model_psnr_x4 = create_swinir_l_x4_real()
safe_load(model_psnr_x4, PSNR_L_X4_PATH)
model_psnr_x4.to(device).eval()

model_gan_x4 = create_swinir_l_x4_real()
safe_load(model_gan_x4, GAN_L_X4_PATH)
model_gan_x4.to(device).eval()

PSNR_MODELS = {2: model_psnr_x4, 4: model_psnr_x4}
GAN_MODELS = {2: model_gan_x4, 4: model_gan_x4}


###############################################################################
# 5) RESIZE INPUT TO ENFORCE SCALE AND OUTPUT CAP
###############################################################################
def resize_input(pil_img: Image.Image, scale: int, max_output_dim=4096):
    w, h = pil_img.size
    target_w, target_h = w * scale, h * scale

    if target_w > max_output_dim or target_h > max_output_dim:
        factor = min(max_output_dim / target_w, max_output_dim / target_h)
        target_w, target_h = int(target_w * factor), int(target_h * factor)

    adjusted_w, adjusted_h = max(1, target_w // 4), max(1, target_h // 4)
    return pil_img.resize((adjusted_w, adjusted_h), Image.BOX)


###############################################################################
# 6) TILE-BASED INFERENCE
###############################################################################
def tile_inference(model: torch.nn.Module, img: torch.Tensor, tile_size=1024, tile_overlap=16):
    b, c, h, w = img.shape
    sf = model.upscale
    out = torch.zeros(b, c, h * sf, w * sf, device=img.device)
    weights = torch.zeros_like(out)

    stride = tile_size - tile_overlap
    h_steps = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
    w_steps = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]

    for yy in h_steps:
        for xx in w_steps:
            patch = img[:, :, yy:yy + tile_size, xx:xx + tile_size]
            patch_out = model(patch)
            out[..., yy * sf:(yy + tile_size) * sf, xx * sf:(xx + tile_size) * sf] += patch_out
            weights[..., yy * sf:(yy + tile_size) * sf, xx * sf:(xx + tile_size) * sf] += 1

    return out / weights.clamp(min=1e-6)


###############################################################################
# 7) COMPRESS OUTPUT UNDER 5MB
###############################################################################
def compress_jpeg_under_5mb(pil_img: Image.Image, max_size=5_000_000, min_quality=30):
    buf = io.BytesIO()
    quality = 95
    while quality >= min_quality:
        buf.seek(0)
        buf.truncate(0)
        pil_img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_size:
            print(f"Final JPEG size={buf.tell()} bytes, quality={quality}")
            buf.seek(0)
            return buf
        quality -= 5
    pil_img.save(buf, format="JPEG", quality=min_quality)
    buf.seek(0)
    return buf


###############################################################################
# 8) ENDPOINT
###############################################################################
@app.post("/upscale")
async def upscale_image(image: UploadFile = File(...), enhance: bool = Form(False), scale: int = Form(4)):
    img_bytes = await image.read()
    pil_input = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    print(f"Original input: {pil_input.size}, scale={scale}, enhance={enhance}")

    pil_input = resize_input(pil_input, scale, max_output_dim=4096)
    np_img = np.array(pil_input, dtype=np.float32) / 255.0
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)
    chosen_model = GAN_MODELS[scale] if enhance else PSNR_MODELS[scale]

    with torch.inference_mode():
        tile_size = 1024
        tile_overlap = 16
        out_tensor = tile_inference(chosen_model, tensor_img, tile_size, tile_overlap)

    out_tensor = torch.clamp(out_tensor, 0, 1)
    out_np = (out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    out_pil = Image.fromarray(out_np)
    jpeg_buf = compress_jpeg_under_5mb(out_pil)

    del tensor_img, out_tensor
    torch.cuda.empty_cache()

    return StreamingResponse(jpeg_buf, media_type="image/jpeg")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=False)
