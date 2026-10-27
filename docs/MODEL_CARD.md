# Model card: QIHGNN-MSCA v0.4.0

> Research prototype. Not a medical device and not for clinical use.

## Overview

A 3D network (8,347,071 parameters, 31.84 MB at fp32) that segments five organ channels from CT or MRI and estimates survival risk from one shared 512-dimensional encoding. It is designed for training on several public datasets that each annotate a single organ.

## Intended use

Research on partially supervised multi-organ segmentation and on joint segmentation and prognosis.

## Not intended for

- Diagnosis, treatment planning or any decision about a patient.
- Reporting on any organ channel that no admitted dataset supervised. The server zeroes those channels before returning a prediction.

## Inputs and outputs

- Input: (B, 4, 128, 128, 64). MRI: T1, T1ce, T2, FLAIR, each z-scored over the head. CT: four Hounsfield windows of one scan. Five covariates for the prognosis head.
- Output: six-channel segmentation (background, brain lesion, liver, kidney, cervix, spleen); log-risk; 1-, 3- and 5-year survival logits; an aleatoric log-variance.

## Training data

Admission is decided by `python -m qihgnn.data.inspection` and recorded in `data/inspection_report.json`.

| Dataset | Channel | Rule |
|---|---|---|
| BraTS 2020 (Kaggle mirror) | brain lesion | admitted when segmentation masks are found; survival data used if present |
| LiTS (Kaggle mirror) | liver | admitted when liver masks are found |
| Cervical cancer segmentation (Kaggle) | cervix | admitted only for volumetric images with masks; otherwise rejected |
| KiTS19 | kidney | not in the current dataset list |
| none | spleen | always unsupervised |

## Evaluation protocol

Per-organ Dice and 95th-percentile Hausdorff distance, only over cases whose dataset supervises the organ and never averaged across channels. Harrell's C-index and time-dependent AUC at 1, 3 and 5 years over cases with survival data. Comparison with a baseline that has the three proposed stages removed, and with each stage removed in turn.

## Results

Segmentation and prognosis: not yet measured. `results/metrics.json` is written by `python -m qihgnn.evaluate`, and the website and API show only what that file contains.

Verified properties of the design: see `results/verification.json` (9 checks).

## Limitations

- No converged run or baseline comparison has been reported yet.
- Each dataset supervises one organ, so adjacent organs in one scan have not been evaluated together.
- The Cox term learns an ordering of risk; calibration depends on the horizon term and real survival data.
- With a softmax head the exclusion loss is identically zero.
- Slices exported as display images lose Hounsfield units and are replicated across input channels.
- Survival covariates are limited to age and extent of resection.

## Ethics and data

Public research datasets only, used under their own licences. No patient data is stored by the platform. The scans on the website are synthetic phantoms unless explicitly marked as held-out public test cases.
