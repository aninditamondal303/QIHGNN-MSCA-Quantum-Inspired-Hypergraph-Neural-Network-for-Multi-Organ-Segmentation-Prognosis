# Decisions log

Every place where this implementation had to choose something the project report leaves open, or where it deviates from the report, with the reason.

## Model

**Layer structure fixed by the parameter budget.** The report gives exact per-stage counts (Table 5.7) but not every layer. The structure was solved so that each stage matches exactly:
- Residual block = conv3×3 (no bias) → affine InstanceNorm → LeakyReLU → conv3×3 (no bias) → affine InstanceNorm, plus a bias-free 1×1×1 projection shortcut. The stem is the same block at stride 1 (4 → 32). Encoder: 3,559,808.
- MPO: two cores (1·16·16·32 + 32·16·16·1 = 16,384) + output bias (256) + affine norm (512) = 17,152.
- Hypergraph: one incidence MLP 256 → 128 → 5 shared by three layers, each with Θ (256 × 256 + bias), LayerNorm and five hyperedge weights = 232,468.
- Multi-scale attention: four self-attention and three cross-attention blocks (each `nn.MultiheadAttention` with a pre-LayerNorm) and a 1×1×1 fusion conv from 1,024 to 256 channels = 2,108,160.
- Decoder: three levels of ConvTranspose (k = 2, bias) + residual block after concatenating the skip, then 1×1×1 to 6 channels = 2,130,214.
- Prognosis: 517 → 256 (LayerNorm, GELU) → 128 (GELU) → 5 = 166,661.

**MPO normalisation.** The report says batch normalisation. With batch size 2 it uses the affine instance norm that the encoder already uses for exactly that reason. The parameter count is identical (2 × 256).

**MPO forward pass.** The cores are contracted into the dense 256 × 256 matrix on each forward pass and applied as a 1×1×1 map. This is algebraically identical to contracting against the input and is cheaper here (65 k multiply-adds per pass). Gradients reach every core.

**The segmentation head reads the shared vector.** The report says both heads read the 512-d representation. Without adding parameters, its two halves act as a per-channel scale and shift (FiLM) on the decoder input.

**Hyperedge weights.** Kept positive with a softplus and initialised to 1.

**Incidence.** Re-estimated from the current node features before each of the three propagation steps (same shared MLP, so no extra parameters).

## Objective

**Exclusion loss under softmax.** Eq. 4.11 penalises foreground mass above 1. With a softmax head the foreground mass is at most 1 by construction, so the term is identically zero. It is implemented as written, reported as a finding, and becomes active with `activation="sigmoid"`.

**An extra verification for the marginal loss.** The report's invariance test moves probability *between unsupervised channels*. Under that test the naive loss only changes when channels become exactly empty (smoothed Dice jumps from about 0 to 1), so it proves invariance but does not show the failure the marginal loss prevents. A second check moves probability *from background into an unlabelled organ* on background-labelled voxels, which is the model correctly recognising an organ its dataset never annotated. The marginal loss is unchanged and the naive loss rises by 0.68. Both checks are in `verify.py`, the tests and the website.

**Covariates.** Five: age (z-scored with fixed population constants so test data never enters normalisation), gross-total resection, subtotal resection, resection unknown, has-clinical-record. Cases without survival data carry zeros and are excluded from the Cox and horizon terms.

## Data

**Inspection before assumption.** Kaggle mirrors repackage data (BraTS as per-slice `.h5`, LiTS as NIfTI or PNG). `qihgnn.data.inspection` records formats, shapes and label values and makes the admission decisions; adapters support each layout the inspection can find.

**CT channels.** A single CT phase is expanded into four Hounsfield windows (abdomen, liver, bone, wide), so CT and four-sequence MRI share the 4-channel input.

**Cervical dataset.** Admitted only with pixel-level masks in volumetric CT or MRI. 2D images would supervise the cervix channel with a modality the network never sees at inference.

**Splits.** Deterministic hash of the patient id (70 / 15 / 15). Slices are grouped into volumes before splitting, and a test asserts no patient is in two splits.

**Brain channel meaning.** BraTS labels tumour sub-regions, not the brain. The channel holds the whole tumour (labels 1, 2 and 4) and is named "brain lesion" in the interface.

## Engineering

**Starlette instead of FastAPI.** FastAPI is a layer over Starlette. The API uses Starlette routes with pydantic models for validation and response schemas, which is the same stack with less magic and was fully testable in the build environment.

**esbuild instead of Vite.** One small build script (`web/scripts/build.mjs`) bundles with esbuild and compiles Tailwind v4 through its JavaScript API, and the dev server proxies `/api`. There are fewer moving parts and the same output on Windows, macOS and Linux.

**Volumes as PNG mosaics.** Browsers decode PNG natively and losslessly, so a 128 × 128 × 64 volume ships as one 1024 × 1024 8-bit PNG per channel. No custom binary format or decoder.

**Synthetic phantoms.** They let the website, the API and CI run before any data is downloaded. They are labelled synthetic everywhere and never used to report performance.

**KaTeX loads lazily.** It is 260 kB, so it is split into its own chunk and loaded the first time an equation is shown.

**Matched baseline.** `configs/baseline.json` turns off the three proposed stages, leaving 5,989,291 parameters. For a strictly parameter-matched comparison, widen the baseline encoder and record that choice here.

## Screening tools (detect, outline, prognosis)

**Why a second set of models.** The 3D research network needs registered 3D volumes in a fixed input space; most images people can actually share are single 2D slices. The screening tools answer the practical questions (is there a tumour, where, how large, what is the outlook) on exactly what a user can upload, and run in the browser so no scan leaves the device.

**Classes.** No tumour, glioma, meningioma, pituitary tumour, and "not a brain MRI". The last class is trained on abdominal CT slices, colposcopy photographs and synthetic non-medical images so the tool rejects inputs it was never meant to read instead of forcing them into a tumour class.

**Duplicate-aware splits.** The Brain Tumor MRI Dataset ships augmented copies ("aug" files) and test images that also appear in its training folder. Images are grouped by a difference hash taken as the minimum over all 8 flips and 90-degree rotations, near matches (at most 2 differing bits) are clustered exactly with a pigeonhole band index, and whole clusters are assigned to train, validation or test. `dedup_audit` in `screen_metrics.json` records how many official test images had a copy in training.

**External test set.** Brain MRI Images for Brain Tumor Detection is used only for testing. It contains every image twice (two folder trees), so it is de-duplicated internally first; then every image whose cluster contains one of our training, validation or test images is removed. Both counts are published.

**BraTS patient IDs.** The per-slice .h5 mirror numbers volumes 1 to 369 (`name_mapping.csv` confirms volume 1 is BraTS20_Training_001). An earlier version assumed 0-based numbering and shifted every survival record by one patient; IDs are now derived from the data (`h5_id_offset`) and covered by a test.

**Calibration and abstention.** A single temperature is fitted on the validation split by minimising negative log-likelihood. The abstention threshold is the smallest confidence at which validation predictions reach 98% accuracy; the coverage and accuracy it gives on the test split are reported.

**Segmentation input.** One grey slice, scaled by the 1st and 99th percentiles of its non-zero pixels. Training draws a random BraTS sequence (T1, T1ce, T2, FLAIR) per sample and mixes in skull-on contrast T1 images, so the outliner is not tied to one sequence or to skull stripping. BraTS splits are per patient; the 2D figshare set has no patient IDs, so its Dice is reported separately with that caveat.

**Volume.** On a 3D NIfTI, every axial slice is outlined and voxels are summed with the file's own voxel size, corrected for the letterbox resize. Its error against expert volumes on held-out BraTS patients is published next to every estimate.

**Prognosis.** A ridge-penalised Cox model (Breslow ties) on age, log whole-tumour volume and extent of resection, fitted on BraTS 2020 survival data. Reported out of sample: repeated 5-fold C-index against age-only, volume-only and age-plus-resection models, Kaplan-Meier curves of out-of-fold risk thirds with a log-rank test, and BraTS short/mid/long class accuracy against the majority-class baseline. 200 bootstrap refits, expressed in the final model's standardisation, give the uncertainty band shown in the browser.

**Browser parity.** The ONNX exports are compared with PyTorch on the same input during export. Preprocessing and the Cox maths are implemented again in TypeScript and unit-tested against fixtures produced by the Python code (`web/tests/screen.test.ts`); agreement is within 1e-5 for normalisation and 1e-9 for survival curves.
