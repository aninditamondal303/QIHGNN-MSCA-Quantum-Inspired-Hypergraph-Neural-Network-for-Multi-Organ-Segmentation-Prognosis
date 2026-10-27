# macOS / Linux shortcuts. On Windows use the VS Code tasks (Terminal > Run Task) or the README commands.
PY ?= .venv/bin/python

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r ml/requirements-dev.txt
	cd web && npm install

verify:
	cd ml && ../$(PY) -m qihgnn.verify && ../$(PY) -m qihgnn.export_web

test:
	$(PY) -m pytest
	cd web && npm test && npm run typecheck

e2e:
	cd web && npm run build && npm run test:e2e

api:
	$(PY) -m uvicorn api.app:app --reload --port 8000

web:
	cd web && npm run dev

smoke:
	cd ml && ../$(PY) -m qihgnn.train --config configs/smoke.json

.PHONY: setup verify test e2e api web smoke
