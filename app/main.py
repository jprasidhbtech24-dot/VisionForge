import base64, io, os, time, threading
from typing import Optional

import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_ID = os.getenv('MODEL_ID', 'stabilityai/sd-turbo')
CLIP_ID = os.getenv('CLIP_ID', 'openai/clip-vit-base-patch32')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

app = FastAPI(title='VisionForge AI', version='1.0.0')
app.mount('/static', StaticFiles(directory='app/static'), name='static')
_pipe = None
_clip_model = None
_clip_processor = None
_lock = threading.Lock()

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    mode: str = Field(default='balanced', pattern='^(fast|balanced|quality)$')
    classical_enhance: bool = True
    seed: Optional[int] = None


def load_generator():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                from diffusers import AutoPipelineForText2Image
                kwargs = {'torch_dtype': DTYPE}
                if DEVICE == 'cuda':
                    kwargs['variant'] = 'fp16'
                _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kwargs)
                _pipe = _pipe.to(DEVICE)
                if DEVICE == 'cuda':
                    try: _pipe.enable_attention_slicing()
                    except Exception: pass
    return _pipe


def load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        with _lock:
            if _clip_model is None:
                from transformers import CLIPModel, CLIPProcessor
                _clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
                _clip_model = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
                _clip_model.eval()
    return _clip_model, _clip_processor


def classical_postprocess(image: Image.Image) -> Image.Image:
    """CLAHE on luminance + mild edge-aware denoising and restrained sharpening."""
    rgb = np.array(image.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=20, sigmaSpace=20)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    sharp = cv2.addWeighted(out, 1.12, blur, -0.12, 0)
    return Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8))


def clip_score(prompt: str, image: Image.Image) -> Optional[float]:
    try:
        model, processor = load_clip()
        inputs = processor(text=[prompt], images=image, return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
        return round(float(score), 4)
    except Exception as exc:
        print('CLIP evaluation unavailable:', repr(exc))
        return None


@app.get('/')
def home():
    return FileResponse('app/static/index.html')

@app.get('/health')
def health():
    return {'status': 'ok', 'device': DEVICE, 'model': MODEL_ID, 'gpu_available': torch.cuda.is_available()}

@app.post('/generate')
def generate(req: GenerateRequest):
    try:
        pipe = load_generator()
        steps = {'fast': 1, 'balanced': 2, 'quality': 4}[req.mode]
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(req.seed))
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            result = pipe(prompt=req.prompt, num_inference_steps=steps, guidance_scale=0.0,
                          generator=generator, height=512, width=512)
        raw = result.images[0]
        generation_ms = round((time.perf_counter() - start) * 1000, 1)
        final = classical_postprocess(raw) if req.classical_enhance else raw
        eval_start = time.perf_counter()
        score = clip_score(req.prompt, final)
        eval_ms = round((time.perf_counter() - eval_start) * 1000, 1)
        buf = io.BytesIO(); final.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('ascii')
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if DEVICE == 'cuda' else None
        return {'image': encoded, 'mime_type': 'image/png', 'mode': req.mode,
                'steps': steps, 'generation_ms': generation_ms, 'clip_score': score,
                'clip_eval_ms': eval_ms, 'peak_gpu_memory_mb': peak_mb,
                'device': DEVICE, 'model': MODEL_ID,
                'classical_postprocess': req.classical_enhance,
                'note': 'Metrics are measured at runtime; latency depends on hardware and model loading is excluded.'}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Generation failed: {str(exc)[:400]}')
