.PHONY: test lint run ui

test:
	python -m pytest -q

lint:
	ruff check .

run:
	uvicorn citerag.api.main:app --reload --port 8000

ui:
	streamlit run src/citerag/ui/app.py
