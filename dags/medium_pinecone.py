# dags/medium_pinecone.py
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator


# -------------------------------
# Config
# -------------------------------
INDEX_NAME = "semantic-search-fast"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRIC = "dotproduct"
DIMENSION = 384

DATA_DIR = Path("/opt/airflow/data")
RAW_CSV = DATA_DIR / "medium_data.csv"
PREPARED_PARQUET = DATA_DIR / "medium_preprocessed.parquet"
INPUT_JSONL = DATA_DIR / "pinecone_input.jsonl"   # explicit "generated input file" for screenshots

CSV_URL = "https://s3-geospatial.s3.us-west-2.amazonaws.com/medium_data.csv"


# -------------------------------
# Utilities
# -------------------------------
def _get_clean_api_key() -> str:
    """Read API key from Airflow Variables and strip any non-ASCII chars/whitespace."""
    raw_key = Variable.get("PINECONE_API_KEY", default_var="")
    api_key = "".join(ch for ch in raw_key if ord(ch) < 128).strip()
    if not api_key:
        raise RuntimeError("Airflow Variable PINECONE_API_KEY not set (or empty).")
    return api_key


# -------------------------------
# DAG
# -------------------------------
default_args = {"owner": "airflow", "retries": 0}

with DAG(
    dag_id="medium_articles_to_pinecone",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pinecone", "embeddings", "semantic-search"],
) as dag:

    # 1) Download CSV
    def download_csv(**_):
        import urllib.request

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[download] url={CSV_URL}")
        print(f"[download] to  ={RAW_CSV}")
        urllib.request.urlretrieve(CSV_URL, RAW_CSV)
        assert RAW_CSV.exists(), "CSV did not download"

        df = pd.read_csv(RAW_CSV)
        size = RAW_CSV.stat().st_size
        print(f"[download] rows={len(df)} size_bytes={size}")
        print("[download] head(3):")
        print(df.head(3).to_string(index=False))

    # 2) Preprocess (make parquet used for embedding)
    def preprocess(**_):
        df = pd.read_csv(RAW_CSV)
        before = len(df)

        # Build metadata as in class slides (title + subtitle)
        df["metadata"] = df.apply(
            lambda r: {
                "title": f"{r.get('title','')} {r.get('subtitle','')}".strip()
            },
            axis=1,
        )

        # Ensure there is an ID
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)

        PREPARED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PREPARED_PARQUET, index=False)

        after = len(df)
        print(f"[preprocess] rows_before={before} rows_after={after}")
        print(f"[preprocess] saved parquet → {PREPARED_PARQUET}")
        print("[preprocess] preview id+metadata:")
        print(df[["id", "metadata"]].head(3).to_string(index=False))

    # 2.5) Generate explicit input file (JSONL) for screenshots
    def prepare_input(**_):
        df = pd.read_parquet(PREPARED_PARQUET)
        records = 0
        with INPUT_JSONL.open("w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                rec = {
                    "id": str(r["id"]),
                    "text": (r["metadata"] or {}).get("title", ""),
                    "metadata": r["metadata"],
                }
                f.write(json.dumps(rec) + "\n")
                records += 1

        print(f"[input] wrote {records} records → {INPUT_JSONL}")
        # show a few lines for the report
        with INPUT_JSONL.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"[input] sample {i+1}: {line.strip()}")

        # Optional directory listing for the screenshot
        print("[input] data folder listing:")
        for p in sorted(DATA_DIR.glob("*")):
            try:
                print("  -", p.name, p.stat().st_size, "bytes")
            except Exception:
                print("  -", p.name)

    # 3) Create Pinecone index
    def create_pinecone_index(**_):
        from pinecone import Pinecone, ServerlessSpec

        api_key = _get_clean_api_key()
        pc = Pinecone(api_key=api_key)
        spec = ServerlessSpec(cloud="aws", region="us-east-1")

        existing = [ix["name"] for ix in pc.list_indexes()]
        print("[index] existing:", existing)

        if INDEX_NAME not in existing:
            print(f"[index] creating {INDEX_NAME} dim={DIMENSION} metric={METRIC}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric=METRIC,
                spec=spec,
            )
        else:
            print(f"[index] {INDEX_NAME} already exists; skipping")

    # 4) Embed + upsert
    def embed_and_upsert(**_):
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        api_key = _get_clean_api_key()

        # Read the parquet produced in preprocess
        df = pd.read_parquet(PREPARED_PARQUET)

        print(f"[embed] loading model: {EMBED_MODEL}")
        model = SentenceTransformer(EMBED_MODEL)

        def emb_from_metadata(md: dict | None) -> list[float]:
            text = (md or {}).get("title", "")
            return model.encode(text).tolist()

        df["values"] = df["metadata"].apply(emb_from_metadata)
        df_upsert = df[["id", "values", "metadata"]].copy()
        df_upsert["id"] = df_upsert["id"].astype(str)

        pc = Pinecone(api_key=api_key)
        index = pc.Index(INDEX_NAME)
        print(f"[upsert] count={len(df_upsert)} → index={INDEX_NAME}")
        index.upsert_from_dataframe(df_upsert)
        print("[upsert] complete; preview:")
        print(df_upsert.head(3).to_string(index=False))

    # 5) Query example (for final screenshot)
    def query_sample(**_):
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer

        api_key = _get_clean_api_key()
        pc = Pinecone(api_key=api_key)
        index = pc.Index(INDEX_NAME)

        model = SentenceTransformer(EMBED_MODEL)
        query = "what is ethics in AI"
        qvec = model.encode(query).tolist()

        print(f"[query] {query}")
        xc = index.query(
            vector=qvec,
            top_k=10,
            include_metadata=True,
            include_values=False,
        )

        matches = xc.get("matches", [])
        if not matches:
            print("[query] no matches returned")
            return

        for i, m in enumerate(matches, start=1):
            title = (m.get("metadata") or {}).get("title")
            print(f"{i:02d}. score={m.get('score'):.4f} id={m.get('id')} title={title}")

    t1 = PythonOperator(task_id="download_csv", python_callable=download_csv)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t2_5 = PythonOperator(task_id="prepare_input", python_callable=prepare_input)   # explicit “generate input file”
    t3 = PythonOperator(task_id="create_pinecone_index", python_callable=create_pinecone_index)
    t4 = PythonOperator(task_id="embed_and_upsert", python_callable=embed_and_upsert)
    t5 = PythonOperator(task_id="query_sample", python_callable=query_sample)

    # Flow
    t1 >> t2 >> t2_5 >> t3 >> t4 >> t5
