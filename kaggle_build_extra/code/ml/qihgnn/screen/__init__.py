"""Clinical screening tools that run on a single uploaded scan (report extension).

detect   : 5-way classifier (no tumour, glioma, meningioma, pituitary, not a brain MRI)
outline  : 2D tumour segmenter applied to one slice or, slice by slice, to a 3D volume
prognosis: Cox model on age, whole-tumour volume and extent of resection (BraTS 2020)

All three are exported to ONNX/JSON and run in the browser, so a scan never leaves the device.
"""
