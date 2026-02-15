from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from rag.core.models import Chunk, RetrievedChunk


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              doc_id TEXT PRIMARY KEY,
              file_path TEXT UNIQUE,
              file_name TEXT,
              file_hash TEXT NOT NULL,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY,
              doc_id TEXT NOT NULL,
              file_name TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              locator TEXT NOT NULL,
              text TEXT NOT NULL,
              metadata_json TEXT,
              FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        self.conn.commit()

    def get_doc_by_path(self, file_path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE file_path = ?", (file_path,)).fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT doc_id, file_path FROM documents").fetchall()

    def upsert_document(self, doc_id: str, file_path: str, file_name: str, file_hash: str) -> None:
        self.conn.execute(
            """
            INSERT INTO documents(doc_id, file_path, file_name, file_hash)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
              doc_id=excluded.doc_id,
              file_name=excluded.file_name,
              file_hash=excluded.file_hash,
              updated_at=CURRENT_TIMESTAMP
            """,
            (doc_id, file_path, file_name, file_hash),
        )
        self.conn.commit()

    def delete_doc_chunks(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        self.conn.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, file_name, chunk_index, locator, text, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (c.chunk_id, c.doc_id, c.file_name, c.chunk_index, c.locator, c.text, json.dumps(c.metadata, ensure_ascii=False))
                for c in chunks
            ],
        )
        self.conn.commit()

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT chunk_id, text FROM chunks ORDER BY rowid ASC").fetchall()

    def get_chunks_by_ids(self, ids: list[str]) -> list[RetrievedChunk]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT chunk_id, text, file_name, locator FROM chunks WHERE chunk_id IN ({placeholders})",
            ids,
        ).fetchall()
        order = {chunk_id: i for i, chunk_id in enumerate(ids)}
        sorted_rows = sorted(rows, key=lambda r: order.get(r["chunk_id"], 999999))
        return [RetrievedChunk(r["chunk_id"], r["text"], 0.0, r["file_name"], r["locator"]) for r in sorted_rows]

    def close(self) -> None:
        self.conn.close()
