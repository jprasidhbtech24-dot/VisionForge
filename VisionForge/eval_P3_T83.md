1RUA24CSE0136,1RUA24CSE0173,1RUA24CSE0166,1RUA24CSE0185
# Evaluation Results — P3 / T83

> Fill this file only with measurements collected from actual runs. Do not submit invented numbers.

## Environment
- Date/time:
- GPU / device:
- CPU and RAM:
- Python / PyTorch / Diffusers versions:
- Model ID and revision:
- Precision:
- Input resolution: 512 × 512
- Warm-up runs:
- Repetitions per prompt/profile:

## Comparative evaluation
| Profile | Steps | Precision | Median generation latency (s) | Peak CUDA allocated memory (MB) | Mean CLIP cosine similarity | Notes |
|---|---:|---|---:|---:|---:|---|
| Quality baseline | 4 | FP16 on CUDA / FP32 CPU | TODO | TODO | TODO | TODO |
| Balanced | 2 | FP16 on CUDA / FP32 CPU | TODO | TODO | TODO | TODO |
| Fast optimized | 1 | FP16 on CUDA / FP32 CPU | TODO | TODO | TODO | TODO |

## Classical CV ablation
| Setting | Median latency (s) | Mean CLIP score | Human visual observations |
|---|---:|---:|---|
| Post-processing OFF | TODO | TODO | TODO |
| Post-processing ON | TODO | TODO | TODO |

## Prompt test set
| ID | Prompt | Seed | Profile | Success / failure | Observation |
|---|---|---:|---|---|---|
| 1 | TODO | TODO | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO | TODO | TODO |

## Failure cases (minimum 3)
1. **Prompt:** TODO  **Observed issue:** TODO  **Likely cause:** TODO  **Mitigation:** TODO
2. **Prompt:** TODO  **Observed issue:** TODO  **Likely cause:** TODO  **Mitigation:** TODO
3. **Prompt:** TODO  **Observed issue:** TODO  **Likely cause:** TODO  **Mitigation:** TODO

## Interpretation
Summarize the measured quality/latency/memory trade-off. State limitations honestly. CLIP cosine similarity is a relative alignment indicator, not a probability. Do not report PSNR/SSIM unless paired ground-truth images exist.
