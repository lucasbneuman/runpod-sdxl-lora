"""
RunPod Serverless Handler for SDXL + Dynamic LoRA Loading
Precaches a base SDXL checkpoint and loads LoRA weights per request.
"""

import os
import base64
import tempfile
import logging
from io import BytesIO
from typing import Any, Dict, Optional

import requests
import torch
import runpod
from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig

logger = logging.getLogger(__name__)

# Environment variables
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "stabilityai/stable-diffusion-xl-base-1.0"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global pipeline (cached on warm worker)
pipe: Optional[StableDiffusionXLPipeline] = None


def load_pipeline() -> StableDiffusionXLPipeline:
    """
    Load base SDXL pipeline on first invocation (warm worker initialization).
    """
    global pipe

    if pipe is not None:
        logger.info(f"[RunPod SDXL LoRA] Pipeline already cached")
        return pipe

    logger.info(f"[RunPod SDXL LoRA] Loading base model: {MODEL_ID}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(DEVICE)

    logger.info(f"[RunPod SDXL LoRA] Model loaded on device: {DEVICE}")
    return pipe


def download_to_temp(url: str, timeout: int = 120) -> str:
    """
    Download a file from presigned URL to temp location.
    Returns path to temp file.
    """
    logger.info(f"[RunPod SDXL LoRA] Downloading LoRA from: {url[:80]}...")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)

    logger.info(f"[RunPod SDXL LoRA] LoRA downloaded to: {path} ({len(r.content)} bytes)")
    return path


def load_and_apply_lora(
    lora_path: str,
    lora_scale: float = 1.0
) -> None:
    """
    Load LoRA weights and apply to pipeline UNet and text encoder.
    Uses PEFT/LoRA library for clean fusing/unfusing.
    """
    global pipe

    logger.info(f"[RunPod SDXL LoRA] Applying LoRA from: {lora_path} (scale={lora_scale})")

    # Option 1: Using diffusers native LoRA loading (if available)
    # This is preferred but may require specific diffusers versions
    try:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        logger.info("[RunPod SDXL LoRA] LoRA fused via diffusers.load_lora_weights()")
        return
    except (AttributeError, NotImplementedError):
        logger.debug("[RunPod SDXL LoRA] diffusers.load_lora_weights not available, trying manual load")

    # Option 2: Manual LoRA state dict loading
    try:
        from safetensors import safe_open

        logger.info("[RunPod SDXL LoRA] Loading LoRA state dict manually")

        # Load LoRA state dict from safetensors
        with safe_open(lora_path, framework="pt", device=DEVICE) as f:
            lora_state_dict = {k: f.get_tensor(k) for k in f.keys()}

        # Apply to UNet and text encoder (simplified approach)
        # In production, you may need to properly match keys and alphas
        _apply_lora_to_model(pipe.unet, lora_state_dict, lora_scale)
        _apply_lora_to_model(pipe.text_encoder, lora_state_dict, lora_scale)

        logger.info("[RunPod SDXL LoRA] LoRA manually applied")

    except Exception as e:
        logger.error(f"[RunPod SDXL LoRA] Failed to load LoRA: {str(e)}")
        raise


def _apply_lora_to_model(
    model: Any,
    lora_state_dict: Dict[str, torch.Tensor],
    lora_scale: float
) -> None:
    """
    Helper to apply LoRA weights to a model.
    This is a simplified version; production may need more sophisticated merging.
    """
    for name, param in model.named_parameters():
        # Look for corresponding LoRA keys in state dict
        lora_up_key = f"{name}.lora_up.weight"
        lora_down_key = f"{name}.lora_down.weight"

        if lora_up_key in lora_state_dict and lora_down_key in lora_state_dict:
            lora_up = lora_state_dict[lora_up_key]
            lora_down = lora_state_dict[lora_down_key]

            # Compute LoRA delta and apply with scale
            # delta = (lora_up @ lora_down) * lora_scale
            delta = (lora_up @ lora_down) * lora_scale
            param.data = param.data + delta.to(param.device).to(param.dtype)


def unfuse_lora() -> None:
    """
    Unfuse LoRA after generation to reset model state.
    Prevents contamination between requests.
    """
    global pipe

    try:
        pipe.unfuse_lora()
        logger.info("[RunPod SDXL LoRA] LoRA unfused")
    except (AttributeError, RuntimeError):
        logger.debug("[RunPod SDXL LoRA] unfuse_lora not available or no LoRA fused")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler.
    Processes generation request with optional LoRA loading.
    """
    global pipe

    try:
        job_id = job.get("id", "unknown")
        inp = job.get("input", {})

        logger.info(f"[RunPod SDXL LoRA] Processing job: {job_id}")

        # Load base model on first request
        pipe = load_pipeline()

        # Extract parameters
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "")
        steps = int(inp.get("steps", 28))
        cfg = float(inp.get("cfg", 5.5))
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))
        seed = inp.get("seed")
        lora_url = inp.get("lora_url")
        lora_scale = float(inp.get("lora_scale", 1.0))

        logger.info(
            f"[RunPod SDXL LoRA] Params: "
            f"prompt={prompt[:50]}..., "
            f"steps={steps}, cfg={cfg}, "
            f"resolution={width}x{height}, "
            f"lora_present={lora_url is not None}"
        )

        # Apply LoRA if provided
        lora_path = None
        if lora_url:
            lora_path = download_to_temp(lora_url)
            load_and_apply_lora(lora_path, lora_scale)

        # Prepare generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Generate image
        logger.info(f"[RunPod SDXL LoRA] Generating image...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        logger.info(f"[RunPod SDXL LoRA] Image generated: {image.size}")

        # Unfuse LoRA to prevent contamination
        if lora_url:
            unfuse_lora()
            # Clean up temp file
            if lora_path and os.path.exists(lora_path):
                os.remove(lora_path)
                logger.info(f"[RunPod SDXL LoRA] Cleaned up temp LoRA: {lora_path}")

        # Encode image as base64
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(f"[RunPod SDXL LoRA] Job {job_id} completed successfully")

        return {
            "image_base64": image_base64,
            "image_size": list(image.size),
        }

    except Exception as e:
        logger.error(f"[RunPod SDXL LoRA] Job failed: {str(e)}", exc_info=True)
        return {
            "error": str(e),
        }


# Start RunPod serverless handler
if __name__ == "__main__":
    logger.info("[RunPod SDXL LoRA] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
