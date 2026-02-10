# RunPod Serverless SDXL + LoRA Worker

Worker de RunPod Serverless para generar imagenes con SDXL y cargar LoRA por URL en cada request.

## Entry point real

- Handler: `rp_handler.py`
- Arranque del contenedor: `Dockerfile` usa `CMD ["python", "-u", "rp_handler.py"]`

## Input esperado

`job["input"]` soporta:

- `prompt` (string, requerido salvo healthcheck)
- `negative_prompt` (string, opcional)
- `steps` (int, default `28`)
- `cfg` (float, default `5.5`)
- `width` (int, default `1024`)
- `height` (int, default `1024`)
- `seed` (int, opcional)
- `lora_url` (string, opcional)
- `lora_scale` (float, default `1.0`)
- `healthcheck` (bool, opcional)

## Healthcheck de deploy

Para las pruebas automaticas de RunPod se responde rapido con:

```json
{
  "input": {
    "healthcheck": true
  }
}
```

## Test local rapido

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecutar:
```bash
python rp_handler.py
```

## Referencias

- RunPod Serverless Overview: https://docs.runpod.io/serverless/overview
- RunPod Serverless Get Started: https://docs.runpod.io/serverless/get-started
