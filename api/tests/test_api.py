"""API tests with Starlette's TestClient."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starlette.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)


def test_health_reports_status_and_disclaimer():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "not for clinical use" in body["disclaimer"]
    assert isinstance(body["inference_ready"], bool)


def test_attention_cost_default_and_validation():
    r = client.get("/api/attention-cost")
    assert r.json()["multi_scale"] == 66_577 and r.json()["naive"] == 1_099_511_627_776
    r = client.get("/api/attention-cost?h=256&w=256&d=128")
    assert r.json()["multi_scale"] == 66_577
    bad = client.get("/api/attention-cost?h=4")
    assert bad.status_code == 422 and "how_to_fix" in bad.json()


def test_mpo_table():
    rows = client.get("/api/mpo").json()["rows"]
    assert {r["bond"]: r["compression"] for r in rows}[32] == 4.0


def test_verification_run_passes_all_checks():
    r = client.post("/api/verification/run")
    assert r.status_code == 200 and r.json()["all_passed"] is True
    assert client.get("/api/verification").json()["parameters"]["total"] == 8_347_071


def test_model_card_lists_labels_and_limitations():
    card = client.get("/api/model-card").json()
    assert [l["name"] for l in card["labels"]][0] == "background"
    assert any("not yet measured" in s for s in card["limitations"])


def test_corpus_marks_unsupervised_channels():
    c = client.get("/api/corpus").json()
    assert "spleen" in c["unsupervisedChannels"]
    assert c["reportCorpus"]["cases"] == 709


def test_cases_listing():
    c = client.get("/api/cases").json()
    assert {x["id"] for x in c["cases"]} >= {"PH-BRAIN-01", "PH-ABDOMEN-01", "PH-PELVIS-01"}


def test_metrics_says_not_yet_measured_without_results():
    metrics = Path(__file__).resolve().parents[2] / "results" / "metrics.json"
    r = client.get("/api/metrics")
    if metrics.exists():
        assert r.status_code == 200
    else:
        assert r.status_code == 404 and r.json()["detail"] == "Not yet measured."


def test_predict_rejects_bad_input():
    assert client.post("/api/predict", json={"case_id": "../etc/passwd"}).status_code == 422
    assert client.post("/api/predict", content=b"not json", headers={"content-type": "application/json"}).status_code == 422


def test_predict_without_model_explains_how_to_fix():
    r = client.post("/api/predict", json={"case_id": "PH-BRAIN-01"})
    if r.status_code == 200:
        assert r.json()["suppressed"]
    else:
        assert r.status_code in (503, 404)
        assert r.json()["how_to_fix"]
