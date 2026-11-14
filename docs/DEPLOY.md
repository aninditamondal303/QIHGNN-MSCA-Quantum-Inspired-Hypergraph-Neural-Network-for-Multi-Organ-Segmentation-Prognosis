# Deploying QIHGNN-MSCA

Three ways to put the project online, from simplest to most complete. All of them use files already in
this repository.

| Option | What visitors get | Cost | Needs |
|---|---|---|---|
| GitHub Pages | The full website in static mode: viewer, architecture, verification lab, corpus, results, literature. "Run network" explains that the API is offline. | Free | A **public** repository |
| Hugging Face Space (Docker) | Website **and** API on one URL. "Run network" works when a checkpoint is in `models/`. | Free (CPU) | A Hugging Face account |
| Render / Railway / Fly.io (Docker) | Same as the Space, on your own subdomain | Free tier or a few dollars | An account on that service |

The site only ever shows scores that `python -m qihgnn.evaluate` wrote, so whichever option you choose,
run the Kaggle training first if you want measured results on the page (see the README).

---

## 1. GitHub Pages (static site)

1. Make the repository public: GitHub → repository → Settings → General → Danger zone → Change visibility.
   (Pages on a private repository needs a paid plan.)
2. Settings → Pages → Build and deployment → Source: **GitHub Actions**.
3. Push to `main`. The `pages` workflow (`.github/workflows/pages.yml`) builds `web/` and publishes it.
4. The address appears in Settings → Pages, normally `https://aninditamondal303.github.io/qihgnn-msca/`.

All asset paths are relative, so the site works under that sub-path without changes.

## 2. Hugging Face Space (website + API, free)

1. Create a Space: huggingface.co → New Space → SDK **Docker** → template **Blank** → hardware **CPU basic**.
2. (Optional) copy a trained checkpoint to `models/qihgnn_full.pt` so "Run network" works.
   Spaces store large files with Git LFS: `git lfs install; git lfs track "models/*.pt"`.
3. Add this block at the very top of the README **in the Space repository only** (Hugging Face reads it;
   keep the GitHub README as it is):

   ```yaml
   ---
   title: QIHGNN-MSCA
   emoji: 🩻
   colorFrom: gray
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

4. Push the repository to the Space's git remote (shown on the Space page). The `Dockerfile` builds the
   website, installs CPU PyTorch and starts the API on port 7860. The first build takes about 10 minutes.

## 3. Render, Railway or Fly.io (Docker)

Point the service at this repository and choose **Dockerfile** as the build method. The image reads the
`PORT` variable these platforms set. Health check path: `/api/health`.

## Run the production image locally

```bash
docker build -t qihgnn-msca .
docker run --rm -p 7860:7860 qihgnn-msca
# open http://localhost:7860
```

With a checkpoint outside the image:

```bash
docker run --rm -p 7860:7860 -v "$PWD/models:/app/models:ro" qihgnn-msca
```

## What is not deployed

- Training data, the preprocessing cache and training runs (large; regenerate with the scripts).
- Any patient scan: the public site shows synthetic phantoms only. If you export real held-out cases with
  `python -m qihgnn.export_web --cases-from-cache N`, check each dataset's licence before publishing them.
