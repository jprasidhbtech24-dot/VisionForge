# 3-minute demo script — VisionForge / T83

**0:00–0:25 — Problem**  
“Our challenge is P3: text-to-image generation under mobile-style compute constraints. High-quality diffusion generation can be expensive in time and memory. We built VisionForge to expose and measure the quality–compute trade-off.”

**0:25–0:55 — Architecture**  
“The user prompt reaches our FastAPI backend. Diffusers runs pretrained SD-Turbo. We offer 1, 2 and 4 denoising-step budgets. The generated image can pass through an OpenCV stage: CLAHE on luminance, bilateral filtering and mild sharpening. CLIP then estimates prompt-image alignment.”

**0:55–1:45 — Live demo**  
“Use a fixed prompt and seed. Generate in Balanced mode, then repeat in Fast mode and Quality mode. Point out that the application measures generation latency and, on CUDA, peak allocated GPU memory. The first run includes model loading, but our reported generation timer excludes model loading.”

**1:45–2:20 — Evaluation**  
“We compare the three step budgets on the same prompt set and hardware. CLIP cosine similarity is a relative alignment indicator, not a probability. We also run an ablation with classical post-processing on and off. We report measured medians rather than claiming unmeasured gains.”

**2:20–3:00 — Limitations and close**  
“Our prototype demonstrates resource-aware inference on a GPU server. It is not yet a native on-device mobile deployment, and cloud GPU memory is not mobile RAM. Lower step counts can trade detail or prompt alignment for speed. Our next step is device-side quantization and benchmarking on a real mobile target.”

## Likely viva questions
- **Why SD-Turbo?** It is a distilled text-to-image model designed for very few inference steps, making it suitable for testing low-step inference budgets.
- **What is optimized?** The inference budget (step count), FP16 on CUDA, and attention slicing where supported. We did not fine-tune weights.
- **What does CLIP score mean?** Cosine similarity between text and image embeddings; useful for relative comparisons, not a calibrated probability or full image-quality metric.
- **Why CLAHE?** It improves local luminance contrast; bilateral filtering smooths while preserving edges; mild sharpening restores perceived crispness. It can also introduce artifacts, so we ablate it.
- **Is it running on a phone?** No. The current deployment is a GPU-backed prototype that explores mobile-relevant efficiency. Native mobile deployment remains future work.
- **Why no PSNR/SSIM?** There is no paired ground-truth image for each arbitrary prompt. Those metrics would be invalid without references.
