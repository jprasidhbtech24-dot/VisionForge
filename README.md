# VisionForge AI — Team T83 (P3)

**Challenge:** Resource-Efficient Text-to-Image Generation for Mobile Devices.

VisionForge is an experimental text-to-image application built around the pretrained **SD-Turbo** diffusion model. It exposes three inference budgets (1, 2 and 4 denoising steps), optional classical image post-processing, runtime CLIP text-image similarity, latency and peak GPU-memory reporting.

> Important research honesty: this hackathon prototype adapts a pretrained model at inference time; it does not claim to have trained or fine-tuned SD-Turbo. Runtime metrics must be measured on the actual judging hardware. Do not report placeholder or invented benchmark numbers.

## Features
- Prompt-conditioned 512×512 generation using `stabilityai/sd-turbo`.
- Fast (1 step), Balanced (2 steps), Quality (4 steps) inference profiles.
- OpenCV classical post-processing: CLAHE on LAB luminance, bilateral filtering and restrained sharpening.
- CLIP cosine similarity as a relative prompt-image alignment indicator (not a calibrated probability).
- Generation latency, CLIP evaluation time and CUDA peak allocated memory when available.
- Responsive web UI and FastAPI backend.

## Run locally / Google Colab
Use a GPU runtime (T4 or better recommended). The first generation downloads model weights and can take several minutes.

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Open `http://localhost:7860`. For Colab, expose port 7860 with the notebook cell in `notebooks/Colab_Setup.ipynb`.

## Deploy to Hugging Face Spaces
1. Create a new **Docker** Space.
2. Upload this repository's files, preserving the `app/` directory.
3. Set hardware to a GPU Space for usable generation speed. CPU fallback exists but may be very slow.
4. Set `MODEL_ID` optionally; default is `stabilityai/sd-turbo`.
5. Open the Space URL and test `/health` and an end-to-end prompt.

The Dockerfile listens on port 7860. Check the model card and license/usage conditions for SD-Turbo and CLIP before public deployment.

## API
`GET /health` returns device/model status.

`POST /generate`
```json
{"prompt":"A red vintage motorcycle parked in a rainy Tokyo alley, cinematic photo","mode":"balanced","classical_enhance":true,"seed":42}
```
Modes: `fast`, `balanced`, `quality`. Response includes base64 PNG, generation time, CLIP score, inference steps and peak CUDA memory (null on CPU).

## Evaluation protocol (run on the same GPU)
- Use the same fixed prompt list and seed list for every profile.
- Warm up once; then run each prompt/profile at least 3 times and report median generation latency.
- Record peak allocated GPU memory, output dimensions, model ID, dtype, step count and hardware.
- Compare CLIP scores as a relative text-image alignment measure. Add human review because CLIP alone does not measure aesthetic quality or factual correctness.
- If a paired reference dataset is available, add FID/KID and human evaluation. Do not claim PSNR/SSIM without paired ground-truth images.
- Ablate classical post-processing on/off using identical prompt and seed; report whether it helps or harms visual quality.
- Include at least 3 failure cases (e.g. complex spatial relations, legible text, counting, hands) with screenshots and analysis.

## Limitations
- SD-Turbo is a distilled text-to-image model; 1/2/4-step profiles are compute-budget comparisons, not a newly trained mobile-native model.
- CUDA memory is not a measurement of total device RAM or mobile memory.
- A GPU-hosted demo is a prototype of resource-aware inference, not proof of on-device mobile deployment. A mobile deployment claim requires testing on a real target device.
- CLIP score varies with prompt wording and should not be interpreted as a probability.
