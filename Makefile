.PHONY: install ui api ingest fetch ask eval test docker clean

install:
	pip install -e ".[dev]"

ui:
	streamlit run app/ui.py

api:
	uvicorn app.api:app --reload

ingest:
	python scripts/scholar.py ingest

fetch:
	python scripts/scholar.py fetch-arxiv "$(Q)" -n 5

ask:
	python scripts/scholar.py ask "$(Q)" --agent

eval:
	python scripts/scholar.py eval --dataset eval/qa_dataset.jsonl --out eval/reports/report.json

test:
	pytest

docker:
	docker compose up --build

clean:
	rm -rf data/chroma data/numpy_store eval/reports **/__pycache__ .pytest_cache
