.PHONY: setup data train serve ui test docker-build docker-run docker-ui clean

setup:
	uv sync --locked

data:
	uv run python prepare_data.py

train:
	uv run python train.py

serve:
	uv run uvicorn predict:app --host 0.0.0.0 --port 9696

ui:
	uv run streamlit run streamlit_app.py

test:
	uv run pytest

docker-build:
	docker build -t rossmann-sales .

docker-run:
	docker run --rm -p 9696:9696 rossmann-sales

docker-ui:
	docker run --rm -p 8501:8501 rossmann-sales \
		streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__
