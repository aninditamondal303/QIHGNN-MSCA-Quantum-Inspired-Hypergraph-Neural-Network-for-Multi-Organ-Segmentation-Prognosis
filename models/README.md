# models/

Put one trained checkpoint here (for example `qihgnn_full.pt`, about 100 MB with optimiser state, or
32 MB weights-only) to ship it with a deployment. The API loads the first `*.pt` file in this folder when
no `runs/LATEST` or `QIHGNN_CHECKPOINT` is set. `.pt` files are git-ignored; use Git LFS or upload the
file to the hosting service directly (see docs/DEPLOY.md).
