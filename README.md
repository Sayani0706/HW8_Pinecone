# Homework 8 — Pinecone + Airflow Semantic Search

This repo contains my Airflow DAG that:
1) Downloads a Medium articles dataset
2) Preprocesses and generates a JSONL input file
3) Creates a Pinecone index (serverless)
4) Embeds titles with `sentence-transformers/all-MiniLM-L6-v2`
5) Upserts to Pinecone
6) Runs a sample semantic search query

## How to run (local Docker)
- `docker compose down`
- `docker compose up -d`
- Open Airflow UI at http://localhost:8081 (login: airflow/airflow)
- Set Airflow Variable: `PINECONE_API_KEY` = your key
- Trigger DAG: `medium_articles_to_pinecone`

## Files
- `dags/medium_pinecone.py` — the DAG
- `docker-compose.yaml` — local Airflow stack
- `screenshots/` — evidence of each step for grading
- `report/` — final report
