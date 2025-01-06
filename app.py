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
# 1) CREATE TWO MODEL CONFIGS (M x2 real, L x4 real)
###############################################################################
def create_swinir_m_x2_real() -> SwinIR:
    """
    'M' real-SwinIR for 2×:
      - embed_dim=180
      - depths=[6,6,6,6,6,6]
      - upsampler='nearest+conv'
      - resi_connection='1conv'
      - scale=2
    """
    model = SwinIR(
        upscale=2,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6,6,6,6,6,6],
        embed_dim=180,
        num_heads=[6,6,6,6,6,6],
        mlp_ratio=2.0,
        upsampler='nearest+conv',
        resi_connection='1conv',
        in_chans=3,
        patch_size=1
    )
    return model

def create_swinir_l_x4_real() -> SwinIR:
    """
    'L' real-SwinIR for 4×:
      - embed_dim=240
      - depths=[6,6,6,6,6,6,6,6,6]
      - upsampler='nearest+conv'
      - resi_connection='3conv'
      - scale=4
    """
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6,6,6,6,6,6,6,6,6],
        embed_dim=240,
        num_heads=[8,8,8,8,8,8,8,8,8],
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
    top_keys = list(ckpt.keys())
    print(f"Loading from {checkpoint_path}, top-level keys: {top_keys}")

    # Attempt recognized subkeys
    for possible_key in ["params", "params_ema", "state_dict", "model"]:
        if possible_key in ckpt:
            model.load_state_dict(ckpt[possible_key], strict=True)
            return
    model.load_state_dict(ckpt, strict=True)


###############################################################################
# 3) CHECKPOINT PATHS
###############################################################################
PSNR_M_X2_PATH = "models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth"
GAN_M_X2_PATH  = "models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth"
PSNR_L_X4_PATH = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth"
GAN_L_X4_PATH  = "models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"


###############################################################################
# 4) CREATE & LOAD MODELS
###############################################################################
model_psnr_x2 = create_swinir_m_x2_real()
safe_load(model_psnr_x2, PSNR_M_X2_PATH)
model_psnr_x2.to(device).eval()

model_gan_x2 = create_swinir_m_x2_real()
safe_load(model_gan_x2, GAN_M_X2_PATH)
model_gan_x2.to(device).eval()

model_psnr_x4 = create_swinir_l_x4_real()
safe_load(model_psnr_x4, PSNR_L_X4_PATH)
model_psnr_x4.to(device).eval()

model_gan_x4 = create_swinir_l_x4_real()
safe_load(model_gan_x4, GAN_L_X4_PATH)
model_gan_x4.to(device).eval()

PSNR_MODELS = {2: model_psnr_x2, 4: model_psnr_x4}
GAN_MODELS  = {2: model_gan_x2,  4: model_gan_x4}


###############################################################################
# 5) TILE-BASED INFERENCE
###############################################################################
def tile_inference(model: torch.nn.Module, img: torch.Tensor, tile_size=512, tile_overlap=32):
    """
    Splits 'img' (B=1, C=3, H, W) into tiles of shape tile_size×tile_size,
    runs the model, and stitches them back to avoid OOM.
    """
    b, c, h, w = img.shape
    sf = model.upscale  # upscaling factor
    out = torch.zeros(b, c, h*sf, w*sf, device=img.device)
    weights = torch.zeros_like(out)

    stride = tile_size - tile_overlap
    # tile steps
    h_steps = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
    w_steps = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]

    for yy in h_steps:
        for xx in w_steps:
            patch = img[:, :, yy:yy+tile_size, xx:xx+tile_size]
            patch_out = model(patch)
            out[..., yy*sf:(yy+tile_size)*sf, xx*sf:(xx+tile_size)*sf] += patch_out
            ones = torch.ones_like(patch_out)
            weights[..., yy*sf:(yy+tile_size)*sf, xx*sf:(xx+tile_size)*sf] += ones

    # average overlap
    out = torch.where(weights == 0, out, out / weights)
    return out


###############################################################################
# 6) HELPER: DOWNSIZE IF INPUT IS TOO LARGE
###############################################################################
def maybe_resize_input(pil_img: Image.Image, scale: int, max_output_dim=4000):
    """
    If the user requests 2× or 4× upscale, we ensure the final dimension won't exceed max_output_dim.
    For instance, if input is 5000×3000 and scale=4 => final 20000×12000 > 8000 => we shrink input.
    """
    w, h = pil_img.size
    final_w = w * scale
    final_h = h * scale

    # If either dimension is above max_output_dim, we resize
    if final_w > max_output_dim or final_h > max_output_dim:
        # e.g. if final_w is 20000, we want <= 8000 => shrink_factor = 8000/20000=0.4
        shrink_factor = min(max_output_dim / final_w, max_output_dim / final_h)
        new_w = int(w * shrink_factor)
        new_h = int(h * shrink_factor)
        print(f"Resizing input from {w}x{h} -> {new_w}x{new_h} so final stays <= {max_output_dim}.")
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img


###############################################################################
# 7) HELPER: COMPRESS OUTPUT UNDER 5MB
###############################################################################
def compress_jpeg_under_5mb(pil_img: Image.Image, max_size=5_000_000, min_quality=30):
    """
    Saves PIL image as JPEG with descending quality until size < max_size or we hit min_quality.
    """
    buf = io.BytesIO()
    quality = 95
    while quality >= min_quality:
        buf.seek(0)
        buf.truncate(0)
        pil_img.save(buf, format="JPEG", quality=quality)
        size = buf.tell()
        if size <= max_size:
            print(f"Final JPEG size={size} bytes, quality={quality}")
            buf.seek(0)
            return buf
        quality -= 5

    print(f"WARNING: Could not compress < {max_size} with quality>={min_quality}. Using lowest.")
    buf.seek(0)
    buf.truncate(0)
    pil_img.save(buf, format="JPEG", quality=min_quality)
    buf.seek(0)
    return buf


###############################################################################
# 8) ENDPOINT
###############################################################################
@app.post("/upscale")
async def upscale_image(
    image: UploadFile = File(...),
    enhance: bool = Form(False),
    scale: int = Form(4)
):
    """
    form-data:
      - 'image': file
      - 'enhance': True=GAN, False=PSNR
      - 'scale': 2 or 4

    Steps:
      1) Possibly shrink input to keep final dimension <= 8000.
      2) If input is large => tile-based inference.
      3) Output as JPEG, forced under 5 MB.
    """
    # Read -> Pillow
    img_bytes = await image.read()
    pil_input = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil_input.size
    print(f"Original input: {w}x{h}, scale={scale}, enhance={enhance}")

    # 1) Maybe shrink input to keep final dimension <= 8000
    pil_input = maybe_resize_input(pil_input, scale, max_output_dim=4000)

    # Convert to float32 [0..1] tensor
    np_img = np.array(pil_input, dtype=np.float32) / 255.0
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)
    chosen_model = GAN_MODELS[scale] if enhance else PSNR_MODELS[scale]

    with torch.inference_mode():
        # 2) Tiling if needed
        # If input dimension is bigger than 2000 => tile
        # (Adjust as you like for speed/memory usage tradeoffs)
        tile_size = 512
        tile_overlap = 32
        _, _, new_h, new_w = tensor_img.shape
        if new_h > 1000 or new_w > 1000:
            print("Performing tile-based inference to keep VRAM usage low...")
            out_tensor = tile_inference(chosen_model, tensor_img, tile_size=tile_size, tile_overlap=tile_overlap)
        else:
            out_tensor = chosen_model(tensor_img)

    # Convert to [0..255], clamp
    out_tensor = torch.clamp(out_tensor, 0, 1)
    out_np = (
        out_tensor.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy() * 255.0
    ).astype(np.uint8)

    out_pil = Image.fromarray(out_np)

    # 3) Compress output to JPEG under 5 MB
    jpeg_buf = compress_jpeg_under_5mb(out_pil, max_size=5_000_000, min_quality=30)

    # Cleanup
    del tensor_img, out_tensor
    torch.cuda.empty_cache()

    return StreamingResponse(jpeg_buf, media_type="image/jpeg")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=False)
