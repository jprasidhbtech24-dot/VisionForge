# Team T83 — GitHub and Hugging Face release checklist

## Before publishing
- Run `python -m compileall app evaluate.py`.
- Install `pip install -r requirements.txt` in a clean Python environment.
- Run `uvicorn app.main:app --host 0.0.0.0 --port 7860`.
- Verify `/health`, then generate at least one image from the browser.
- Run `python evaluate.py --url http://127.0.0.1:7860 --repeats 3` and commit the generated `evaluation_results/` only after inspecting the measurements.
- Never commit access tokens, private keys, or `.env` files.

## GitHub (run locally; authenticate with GitHub CLI)
```bash
gh auth login
cd visionforge
git init
git add .
git commit -m "Initial VisionForge P3 implementation"
gh repo create VisionForge-T83-P3 --public --source=. --remote=origin --push
```
If the repository already exists, use its URL with `git remote add origin <repo-url>` and `git push -u origin main`.

## Hugging Face Spaces (run locally; authenticate with HF CLI)
1. Create a new Space at https://huggingface.co/new-space .
2. Choose **Docker** SDK and a GPU hardware tier if available.
3. From the repository root, install/login using `pip install -U huggingface_hub && hf auth login`.
4. Create a Space named `visionforge-t83` under your account/organization, then run:
```bash
hf repo create <username>/visionforge-t83 --repo-type space --space-sdk docker
# Add the Space as a git remote (use the credential helper; never paste a token into source files).
git remote add hf https://huggingface.co/spaces/<username>/visionforge-t83
git push hf main
```
5. Set the Space hardware to GPU if available; wait for build, open the Space, check `/health`, and generate an image.

## Release evidence
Record the public GitHub URL, public Space URL, commit hash, Space hardware, model ID, test date, and attach raw evaluation CSV plus screenshots. Do not claim deployment succeeded until the public URL has been opened and an end-to-end generation completed.
