.PHONY: train dev lint test test-unit test-integration

train:
	python train/train_baseline.py

dev:
	uvicorn inference.main:app --reload --port 8010

lint:
	ruff check .

test:
	pytest -q

test-unit:
	pytest -q tests/unit

test-integration:
	pytest -q tests/integration
