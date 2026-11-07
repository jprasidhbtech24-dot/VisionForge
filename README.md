# VisionForge AI — Team T83 (P3)

**Challenge:** Resource-Efficient Text-to-Image Generation for Mobile Devices.

VisionForge is an experimental text-to-image application built around the pretrained **SD-Turbo** diffusion model. It exposes three inference budgets (1, 2 and 4 denoising steps), an automatic resource-aware profile, optional classical image post-processing, runtime CLIP text-image similarity, latency, CPU process RAM snapshots and peak generation GPU memory.

> Important research honesty: this hackathon prototype adapts a pretrained model at inference time; it does not claim to have trained or fine-tuned SD-Turbo. Runtime metrics must be measured on the actual judging hardware. Do not report placeholder or invented benchmark numbers.

## Features
- Prompt-conditioned 512×512 generation using `stabilityai/sd-turbo`.
- Fast (1 step), Balanced (2 steps), Quality (4 steps) inference profiles.
- Auto profile chooses an inference budget and caps output resolution from currently available CPU RAM or free GPU memory; the UI explains its choice.
- Same-prompt comparison of a 4-step baseline, 1-step optimized run and adaptive Auto run, using the same seed and requested resolution.
- OpenCV classical post-processing: CLAHE on LAB luminance, bilateral filtering and restrained sharpening.
- CLIP cosine similarity as a relative prompt-image alignment indicator (not a calibrated probability).
- Generation latency, CLIP evaluation time, CUDA peak allocated memory when available, and before/after process RAM snapshots.
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
`GET /health` returns device/model status and current resource availability.

`POST /generate`
```json
{"prompt":"A red vintage motorcycle parked in a rainy Tokyo alley, cinematic photo","mode":"balanced","classical_enhance":true,"seed":42,"resolution":512}
```
Modes: `fast`, `balanced`, `quality`, `auto`. Resolutions: 256, 384 or 512 square. Response includes a base64 PNG, generation time, CLIP score, selected mode/size, inference steps, peak CUDA allocated memory (null on CPU), and CPU process/system RAM snapshots. `POST /compare` accepts the same request and returns the baseline/optimized/Auto results sequentially with the same prompt and seed.

## Evaluation protocol (run on the same GPU)
- Use the same fixed prompt list and seed list for every profile.
- Warm up once; then run each prompt/profile at least 3 times and report median generation latency.
- Record peak allocated GPU memory, CPU process RAM snapshots, output dimensions, model ID, dtype, step count and hardware.
- Compare CLIP scores as a relative text-image alignment measure. Add human review because CLIP alone does not measure aesthetic quality or factual correctness.
- If a paired reference dataset is available, add FID/KID and human evaluation. Do not claim PSNR/SSIM without paired ground-truth images.
- Ablate classical post-processing on/off using identical prompt and seed; report whether it helps or harms visual quality.
- Include at least 3 failure cases (e.g. complex spatial relations, legible text, counting, hands) with screenshots and analysis.

## Limitations
- SD-Turbo is a distilled text-to-image model; 1/2/4-step profiles are compute-budget comparisons, not a newly trained mobile-native model.
- CUDA memory is not a measurement of total device RAM or mobile memory. CPU process RAM fields are point-in-time snapshots, not peak resident memory.
- Auto uses a transparent heuristic based on current free GPU memory or available system RAM. It is a resource-aware server demo, not a mobile-device detector or proof of on-device deployment.
- A GPU-hosted demo is a prototype of resource-aware inference, not proof of on-device mobile deployment. A mobile deployment claim requires testing on a real target device.
- CLIP score varies with prompt wording and should not be interpreted as a probability.
