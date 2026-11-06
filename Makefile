.PHONY: setup data train serve test docker-build docker-run clean

setup:
	uv sync

data:
	uv run python prepare_data.py

train:
	uv run python train.py

serve:
	uv run uvicorn predict:app --host 0.0.0.0 --port 9696

test:
	uv run pytest

docker-build:
	docker build -t rossmann-sales .

docker-run:
	docker run --rm -p 9696:9696 rossmann-sales

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__
