"""
RunPod Serverless Handler for SDXL + Dynamic LoRA Loading.
Caches a base SDXL checkpoint and applies LoRA per request.
"""

import base64
import logging
import os
import sys
import tempfile
from io import BytesIO
from typing import Any, Dict, Optional

import requests
import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
pipe: Optional[Any] = None
torch_mod: Optional[Any] = None


def get_torch() -> Any:
    """Lazy-import torch to keep startup fast and resilient."""
    global torch_mod
    if torch_mod is None:
        import torch as _torch

        torch_mod = _torch
    return torch_mod


def get_device_dtype() -> Any:
    """Resolve execution device and dtype lazily."""
    torch = get_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def load_pipeline() -> Any:
    """Load and cache base SDXL model once per worker."""
    global pipe

    if pipe is not None:
        logger.info("Pipeline already cached")
        return pipe

    device, dtype = get_device_dtype()
    from diffusers import StableDiffusionXLPipeline

    logger.info("Loading base model: %s", MODEL_ID)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    logger.info("Model loaded on device=%s dtype=%s", device, dtype)
    return pipe


def download_to_temp(url: str, timeout: int = 120) -> str:
    """Download LoRA weights from presigned URL and return local temp path."""
    logger.info("Downloading LoRA from URL")

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    try:
        with requests.get(url, stream=True, timeout=(20, timeout)) as resp:
            resp.raise_for_status()
            with os.fdopen(fd, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        logger.info("LoRA downloaded: %s", path)
        return path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(path):
            os.remove(path)
        raise


def apply_lora(lora_path: str, lora_scale: float) -> None:
    """Load and fuse LoRA into currently cached pipeline."""
    global pipe
    if pipe is None:
        raise RuntimeError("Pipeline is not loaded.")

    logger.info("Applying LoRA path=%s scale=%.3f", lora_path, lora_scale)
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=lora_scale)
    logger.info("LoRA fused")


def cleanup_lora() -> None:
    """Revert LoRA changes to avoid state leakage between requests."""
    global pipe
    if pipe is None:
        return

    try:
        pipe.unfuse_lora()
        logger.info("LoRA unfused")
    except Exception:
        logger.info("No fused LoRA to unfuse")

    # For compatibility across diffusers versions.
    unload = getattr(pipe, "unload_lora_weights", None)
    if callable(unload):
        try:
            unload()
            logger.info("LoRA weights unloaded")
        except Exception:
            logger.info("LoRA weights unload skipped")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod request handler."""
    global pipe

    lora_path: Optional[str] = None
    lora_applied = False

    try:
        job_id = job.get("id", "unknown")
        inp = job.get("input", {})
        logger.info("Processing job id=%s", job_id)

        # Fast path for deployment tests / liveness checks.
        if bool(inp.get("healthcheck")):
            return {
                "status": "ok",
                "model_cached": pipe is not None,
            }

        pipe = load_pipeline()

        prompt = str(inp.get("prompt", "")).strip()
        if not prompt:
            return {
                "status": "invalid_input",
                "message": "Missing required field: prompt",
            }

        negative_prompt = str(inp.get("negative_prompt", ""))
        steps = int(inp.get("steps", 28))
        cfg = float(inp.get("cfg", 5.5))
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))
        seed = inp.get("seed")
        lora_url = inp.get("lora_url")
        lora_scale = float(inp.get("lora_scale", 1.0))

        logger.info(
            "Params steps=%s cfg=%s size=%sx%s lora=%s",
            steps,
            cfg,
            width,
            height,
            bool(lora_url),
        )

        if lora_url:
            lora_path = download_to_temp(str(lora_url))
            apply_lora(lora_path, lora_scale)
            lora_applied = True

        generator = None
        if seed is not None:
            torch = get_torch()
            device, _ = get_device_dtype()
            generator = torch.Generator(device=device).manual_seed(int(seed))

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        buf = BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info("Job id=%s completed", job_id)
        return {
            "image_base64": image_base64,
            "image_size": list(image.size),
        }
    except Exception as exc:
        logger.exception("Job failed: %s", exc)
        return {"error": str(exc)}
    finally:
        if lora_applied:
            cleanup_lora()
        if lora_path and os.path.exists(lora_path):
            try:
                os.remove(lora_path)
            except OSError:
                logger.warning("Failed to remove temp LoRA: %s", lora_path)


if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({"handler": handler})
