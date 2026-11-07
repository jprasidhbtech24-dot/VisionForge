# VisionForge AI — Design Document

## 1. Problem and objective
P3 asks for a lightweight text-to-image system that preserves semantic alignment and visual quality while reducing inference compute, memory and latency for constrained devices. VisionForge demonstrates budget-aware inference using a pretrained distilled diffusion model.

## 2. Users and use cases
A user enters a natural-language prompt, selects a compute profile, optionally enables classical post-processing, and receives a generated image plus measured runtime indicators. A judge can repeat a prompt with a fixed seed to compare profiles.

## 3. System architecture
- **Presentation:** responsive HTML/CSS/JavaScript interface.
- **API:** FastAPI validates prompt, profile and seed.
- **Generation:** Diffusers loads SD-Turbo once; inference profile selects 1, 2 or 4 steps.
- **Classical CV:** OpenCV LAB luminance CLAHE, bilateral filtering and restrained sharpening.
- **Evaluation:** CLIP image/text embeddings; cosine similarity is returned as a relative alignment score.
- **Telemetry:** wall-clock generation latency and CUDA peak allocated memory.

## 4. Model and adaptation strategy
Base model: `stabilityai/sd-turbo` through Hugging Face Diffusers. Adaptation is inference-time compute budgeting (step count and device precision), not weight fine-tuning. The model is pretrained and distilled; disclose this accurately in the demo and report.

## 5. Optimization strategy
1. Reduced denoising steps: 4-step quality, 2-step balanced, 1-step fast.
2. Reduced precision: FP16 on CUDA; FP32 CPU fallback.
3. Attention slicing is enabled where supported.

These optimizations must be evaluated empirically. Lower step counts can reduce latency but may reduce prompt adherence or image detail.

## 6. Classical computer vision
Optional post-processing uses CLAHE on LAB luminance for local contrast, bilateral filtering to smooth while retaining edges, and a mild unsharp blend. The ablation compares this stage enabled and disabled. It is a post-processing baseline, not a standalone text-to-image generator.

## 7. Evaluation plan
Report median generation latency across repeated runs, CUDA peak allocated memory, model identifier, precision, step count and CLIP cosine similarity. Use fixed prompts/seeds and same hardware. Add human assessment and FID/KID only when a suitable evaluation set is available. Do not use PSNR/SSIM without paired ground truth.

## 8. Data
No custom training dataset is required for this inference prototype. Evaluation prompts should be curated to cover single objects, multiple objects, spatial relations, styles, lighting, text rendering and uncommon concepts. Store prompt list and seeds in the repository for reproducibility.

## 9. Deployment and reproducibility
Docker image serves the FastAPI API and static frontend on port 7860. Deploy as a Docker Space on Hugging Face. A GPU Space is recommended. Record hardware tier, software versions, model revision and date in results.

## 10. Risks, limitations and ethics
- GPU cloud inference does not prove native mobile deployment.
- SD-Turbo/CLIP license and model-card restrictions must be checked before public use.
- Generated content can be inaccurate, biased or misleading.
- CLIP similarity is not a probability or complete quality metric.
- Results vary by GPU, load, precision and warm-up state.
- Do not claim fine-tuning, mobile benchmark results or improvements unless actually measured.

## 11. Failure analysis template
For each failure record: prompt, seed, profile, output screenshot, observed failure, likely cause, whether post-processing worsened it, and proposed mitigation. Include at least three real observed cases.
