# Model card: QIHGNN-MSCA

> Research and education only. Not a medical device, not cleared by any regulator, and not for diagnosis or treatment.

Author: Anindita Mondal.

## Screening tools (run in the browser)

| Model | Task | Architecture | Training data | Held-out result |
|---|---|---|---|---|
| Brain classifier | No tumour, glioma, meningioma, pituitary tumour, not a brain MRI | EfficientNet-B0, ImageNet-initialised, 4.0 M parameters, temperature 0.79 | Brain Tumor MRI Dataset, BraTS 2020 slices, abdominal CT, cervical cell images, synthetic images; split by near-duplicate cluster or patient | accuracy 97.4%, tumour sensitivity 98.4%, specificity 98.8%, ECE 1.2%; outside data: sensitivity 75.9%, AUC 0.986 |
| Brain segmenter | Whole tumour (tumour core and oedema) on one grey slice | 2D U-Net, 1.9 M parameters | BraTS 2020 (all four sequences, patient split) and skull-on contrast T1 slices | 3D Dice 0.749 (T2), volume error median 8.9 cm³ |
| Prognosis | Survival curve for glioma | Ridge Cox model on age, log tumour volume, extent of resection; 200 bootstrap refits | BraTS 2020 survival data, 236 patients | C-index 0.634 (age alone 0.621) |
| Liver segmenter | Background, liver, tumour on a CT slice | Two stages: 2D U-Net for the liver (3 classes), second U-Net on a crop around the liver for tumours | LiTS (131 patients, patient split, 25 test patients); JPEG mask specks removed | liver whole-scan Dice 0.93; tumour figures in `organs_metrics.json` once the second stage is published |
| Cytology segmenter | Background, cytoplasm, nucleus on a microscope image of cervical cells | 2D U-Net, 3 classes, RGB, 1.9 M parameters | Cervical cytology, 686 images with cell and nucleus outlines, slide split (100 test images) | nucleus Dice 0.751, cell Dice 0.877; nuclei count median error 5 (r 0.904); median N:C error 0.004 (r 0.928) |

### Intended use

Demonstration and education: what a scan analysis pipeline can and cannot do, with its accuracy measured on held-out and outside data. Images are processed in the user's browser and never uploaded.

### Not intended for

Diagnosis, screening of real patients, treatment planning, or any decision about a person's care.

### Known limitations

- Public datasets do not represent every scanner, protocol, population or tumour type; metastases, lymphoma and other tumours are not among the classes.
- On an independent dataset the classifier finds fewer tumours at its default threshold (75.9% vs 98.4% in-distribution) while its ranking stays good (AUC 0.986); real-world performance can be lower still. Only 2 tumour-free outside images survived de-duplication, so outside specificity is unmeasured.
- A single 2D slice can miss a tumour elsewhere in the brain or liver.
- The brain outline covers the whole tumour region including oedema and does not separate sub-regions.
- The prognosis model ranks risk modestly better than age alone and ignores treatment, molecular markers (such as IDH and MGMT) and performance status. Its survival-class accuracy equals the majority-class baseline.
- Liver training images were already windowed for display and carry no voxel size; Hounsfield-unit uploads are windowed at -160 to 240 HU.
- The cytology data has no diagnosis labels: the tool measures nucleus-to-cytoplasm ratios and flags outliers, and does not grade the cells.

### Evaluation design

Group-level splits (patient, CT volume, slide, or near-duplicate cluster under flips and rotations), an outside test set de-duplicated internally and against all of our data, Wilson confidence intervals, calibration error, abstention coverage, and parity tests between PyTorch, ONNX and the TypeScript re-implementation. Details in `docs/DECISIONS.md`.

## Research model: 3D QIHGNN-MSCA

### Overview

A 3D network (8,347,071 parameters, 31.84 MB at fp32) that segments five organ channels from CT or MRI and estimates survival risk from one shared 512-dimensional encoding. It is designed for training on several public datasets that each annotate a single organ.

### Intended use

Research on partially supervised multi-organ segmentation and on joint segmentation and prognosis.

### Not intended for

- Diagnosis, treatment planning or any decision about a patient.
- Reporting on any organ channel that no admitted dataset supervised. The server zeroes those channels before returning a prediction.

### Inputs and outputs

- Input: (B, 4, 128, 128, 64). MRI: T1, T1ce, T2, FLAIR, each z-scored over the head. CT: four Hounsfield windows of one scan. Five covariates for the prognosis head.
- Output: six-channel segmentation (background, brain lesion, liver, kidney, cervix, spleen); log-risk; 1-, 3- and 5-year survival logits; an aleatoric log-variance.

### Training data

Admission is decided by `python -m qihgnn.data.inspection` and recorded in `data/inspection_report.json`.

| Dataset | Channel | Rule |
|---|---|---|
| BraTS 2020 (Kaggle mirror) | brain lesion | admitted when segmentation masks are found; survival data used if present |
| LiTS (Kaggle mirror) | liver | admitted when liver masks are found |
| Cervical cancer segmentation (Kaggle) | cervix | admitted only for volumetric images with masks; otherwise rejected |
| KiTS19 | kidney | not in the current dataset list |
| none | spleen | always unsupervised |

### Evaluation protocol

Per-organ Dice and 95th-percentile Hausdorff distance, only over cases whose dataset supervises the organ and never averaged across channels. Harrell's C-index and time-dependent AUC at 1, 3 and 5 years over cases with survival data. Comparison with a baseline that has the three proposed stages removed, and with each stage removed in turn.

### Results

Segmentation and prognosis: not yet measured. `results/metrics.json` is written by `python -m qihgnn.evaluate`, and the website and API show only what that file contains.

Verified properties of the design: see `results/verification.json` (9 checks).

### Limitations

- No converged run or baseline comparison has been reported yet.
- Each dataset supervises one organ, so adjacent organs in one scan have not been evaluated together.
- The Cox term learns an ordering of risk; calibration depends on the horizon term and real survival data.
- With a softmax head the exclusion loss is identically zero.
- Slices exported as display images lose Hounsfield units and are replicated across input channels.
- Survival covariates are limited to age and extent of resection.

### Ethics and data

Public research datasets only, used under their own licences. No patient data is stored by the platform. The scans on the website are synthetic phantoms unless explicitly marked as held-out public test cases.
