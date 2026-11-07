# VisionForge architecture

```mermaid
flowchart TD
    U[User / Browser] --> UI[React-free responsive Web UI<br/>HTML CSS JavaScript]
    UI --> API[FastAPI /generate]
    API --> CFG[Inference profile selector<br/>1 / 2 / 4 steps]
    CFG --> DIFF[Diffusers pipeline]
    DIFF --> SD[Pretrained SD-Turbo]
    SD --> RAW[Generated RGB image]
    RAW --> CV{Classical CV enabled?}
    CV -->|Yes| PP[OpenCV CLAHE + bilateral + sharpening]
    CV -->|No| IMG[Final image]
    PP --> IMG
    IMG --> CLIP[CLIP text-image similarity]
    IMG --> TEL[Latency / CUDA peak memory]
    CLIP --> RESP[JSON response + PNG]
    TEL --> RESP
    RESP --> UI
```

## Data flow
1. Browser submits prompt, mode, seed and post-processing flag.
2. API validates input and chooses denoising step budget.
3. Diffusers generates a 512×512 image using SD-Turbo.
4. Optional OpenCV post-processing is applied.
5. CLIP cosine similarity and runtime metrics are calculated.
6. API returns a PNG and metrics to the browser.
