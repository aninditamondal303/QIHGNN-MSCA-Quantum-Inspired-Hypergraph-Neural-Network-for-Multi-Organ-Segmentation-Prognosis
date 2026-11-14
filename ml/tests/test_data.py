"""Data pipeline: preprocessing, label handling, splits, survival parsing, inspection rules, phantoms."""
from pathlib import Path

import numpy as np
from PIL import Image

from qihgnn.config import LABELS
from qihgnn.data import preprocess as P
from qihgnn.data.adapters import _slice_stacks
from qihgnn.data.inspection import decide, inspect_dir
from qihgnn.data.splits import assert_disjoint, assign, split_of
from qihgnn.data.survival import parse_days, read_survival_csv
from qihgnn.export_web import mosaic
from qihgnn.phantoms import all_phantoms
from qihgnn.serving import label_map, suppress


def test_ct_window_range_and_channels():
    hu = np.array([[[-1000.0, 40.0, 3000.0]]])
    ch = P.ct_to_channels(hu)
    assert ch.shape == (4, 1, 1, 3)
    assert ch.min() >= 0 and ch.max() <= 1
    assert ch[0, 0, 0, 0] == 0.0 and ch[0, 0, 0, 2] == 1.0


def test_mri_zscore_ignores_background():
    v = np.zeros((4, 4, 4)); v[1:3, 1:3, 1:3] = np.arange(8).reshape(2, 2, 2) + 10
    z = P.mri_zscore(v)
    assert z[0, 0, 0] == 0
    assert abs(z[v != 0].mean()) < 1e-6 and abs(z[v != 0].std() - 1) < 1e-5


def test_display_masks_are_rescaled_from_255():
    m = np.array([[0, 255], [255, 0]], np.uint8)
    assert P.rescale_display_mask(m).tolist() == [[0, 1], [1, 0]]
    assert P.rescale_display_mask(np.array([[0, 1, 2]])).tolist() == [[0, 1, 2]]


def test_remap_and_supervision_vector():
    lab = np.array([0, 1, 2, 4, 3])
    out = P.remap_labels(lab, {1: "brain", 2: "brain", 4: "brain"})
    assert out.tolist() == [0, 1, 1, 1, 0]
    v = P.supervision_vector(["liver"])
    assert v.tolist() == [True, False, True, False, False, False]


def test_to_grid_exact_shape():
    vol = np.random.rand(4, 97, 113, 41).astype(np.float32)
    assert P.to_grid(vol, (128, 128, 64), order=1).shape == (4, 128, 128, 64)
    lab = (np.random.rand(97, 113, 41) > 0.5).astype(np.uint8)
    out = P.to_grid(lab, (128, 128, 64), order=0)
    assert out.shape == (128, 128, 64) and set(np.unique(out)) <= {0, 1}


def test_patient_level_split_is_deterministic_and_disjoint():
    ids = [f"BraTS20_Training_{i:03d}" for i in range(400)]
    s = assign(ids)
    assert_disjoint(s)
    assert sum(len(v) for v in s.values()) == 400
    assert split_of("abc") == split_of("abc")
    assert 0.6 < len(s["train"]) / 400 < 0.8


def test_slices_of_one_patient_share_a_split():
    # The split key is the patient id, never the slice id.
    assert {split_of("LiTS_007") for _ in range(10)} == {split_of("LiTS_007")}


def test_survival_parsing(tmp_path: Path):
    assert parse_days("455") == (455.0, 1)
    assert parse_days("ALIVE (361 days later)") == (361.0, 0)
    assert parse_days("") == (None, 0)
    csv = tmp_path / "survival_info.csv"
    csv.write_text("Brats20ID,Age,Survival_days,Extent_of_Resection\nBraTS20_Training_001,60.463,289,GTR\n"
                   "BraTS20_Training_002,52.263,ALIVE (361 days later),\nBraTS20_Training_003,54.301,,STR\n")
    recs = read_survival_csv(csv)
    assert set(recs) == {"BraTS20_Training_001", "BraTS20_Training_002"}
    assert recs["BraTS20_Training_001"].event == 1 and recs["BraTS20_Training_002"].event == 0
    assert recs["BraTS20_Training_001"].covariates[1] == 1.0
    assert recs["BraTS20_Training_002"].covariates[3] == 1.0


def _write_slices(root: Path, patients=2, slices=10):
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    for p in range(patients):
        for s in range(slices):
            img = (np.random.rand(32, 32) * 255).astype(np.uint8)
            msk = np.zeros((32, 32), np.uint8); msk[8:20, 8:20] = 255
            Image.fromarray(img).save(root / "images" / f"{p}_{s}.png")
            Image.fromarray(msk).save(root / "masks" / f"{p}_{s}.png")


def test_inspection_rejects_2d_cervical_images_with_masks(tmp_path: Path):
    _write_slices(tmp_path / "cervical")
    summary = inspect_dir(tmp_path / "cervical")
    assert summary["mask_like_files"] == 20
    dec = decide("cervical", summary)
    assert dec["status"] == "rejected" and "2D" in dec["reason"]


def test_inspection_rejects_cervical_without_masks(tmp_path: Path):
    (tmp_path / "c").mkdir()
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "c" / "img1.png")
    assert decide("cervical", inspect_dir(tmp_path / "c"))["status"] == "rejected"


def test_inspection_pending_when_missing(tmp_path: Path):
    assert decide("lits", inspect_dir(tmp_path / "nope"))["status"] == "pending inspection"


def test_slice_stack_adapter_rebuilds_volumes_and_rescales_masks(tmp_path: Path):
    _write_slices(tmp_path / "lits", patients=2, slices=10)
    cases = list(_slice_stacks(tmp_path / "lits", "lits", "liver"))
    assert len(cases) == 2
    c = cases[0]
    assert c.image.shape == (4, 32, 32, 10)
    assert set(np.unique(c.label)) == {0, LABELS.index("liver")}
    assert c.supervised.tolist() == [True, False, True, False, False, False]


def test_suppression_never_returns_an_unsupervised_channel():
    probs = np.zeros((6, 4)); probs[4] = 0.9; probs[0] = 0.1  # cervix is the argmax everywhere
    assert (label_map(probs, ["cervix", "spleen"]) != LABELS.index("cervix")).all()
    assert suppress(probs, ["cervix"])[4].sum() == 0
    try:
        suppress(probs, ["background"])
        raise AssertionError("background must not be suppressible")
    except ValueError:
        pass


def test_phantoms_have_expected_structure():
    for ph in all_phantoms():
        assert ph.image.shape == (4, 128, 128, 64)
        assert ph.label.shape == (128, 128, 64)
        assert 0 <= ph.image.min() and ph.image.max() <= 1
        for c in np.unique(ph.label):
            assert c < len(LABELS)
    brain, abdomen, pelvis = all_phantoms()
    assert (brain.label == LABELS.index("brain")).sum() > 1000
    assert (abdomen.label == LABELS.index("liver")).sum() > 10000
    assert not pelvis.supervised[LABELS.index("cervix")]


def test_mosaic_layout_round_trips():
    vol = np.random.randint(0, 255, (16, 12, 10)).astype(np.uint8)
    m, cols = mosaic(vol)
    assert cols == 4 and m.shape == (3 * 16, 4 * 12)
    k = 7
    r, c = divmod(k, cols)
    assert (m[r * 16:(r + 1) * 16, c * 12:(c + 1) * 12] == vol[:, :, k]).all()


def test_jpeg_display_mask_ignores_compression_ringing():
    m = np.zeros((8, 8), np.uint8)
    m[2:6, 2:6] = 255
    m[1, 2:6] = 9  # JPEG ringing just outside the organ edge
    m[6, 2:6] = 31
    m[3, 3] = 240  # compression noise inside the organ
    out = P.rescale_display_mask(m)
    assert out.sum() == 16 and out[1].sum() == 0 and out[6].sum() == 0


def test_list_files_matches_rglob(tmp_path):
    from qihgnn.data.listing import list_files

    for rel in ("a/x.h5", "a/b/y.H5", "c/z.jpg", "top.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"0")
    assert sorted(list_files(tmp_path)) == sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert [p.name for p in list_files(tmp_path, (".h5",))] == ["x.h5", "y.H5"]


def test_brats_h5_ids_follow_the_mirror_numbering():
    from qihgnn.data.adapters import brats_h5_patient_id, h5_id_offset

    assert h5_id_offset({1: 0, 2: 0, 369: 0}) == 0 and brats_h5_patient_id(1, 0) == "BraTS20_Training_001"
    assert h5_id_offset({0: 0, 368: 0}) == 1 and brats_h5_patient_id(368, 1) == "BraTS20_Training_369"


def test_label_coded_jpeg_masks_survive():
    # LiTS mirror masks store 0/1/2 (and JPEG noise 3) directly; they must not be thresholded away.
    m = np.array([[0, 1, 1], [2, 3, 0]], np.uint8)
    out = P.rescale_display_mask(m)
    assert out.tolist() == m.tolist() and (out > 0).sum() == 4
