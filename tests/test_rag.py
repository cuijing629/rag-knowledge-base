from unittest.mock import Mock

import pymupdf
from fastapi.testclient import TestClient

from backend.documents import parse_pdf, split_pages
from backend.main import Draft, Settings, create_app


def make_pdf():
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page().insert_text((50, 50), "At least two reviewers.")
        return pdf.tobytes()


def test_pdf_preserves_page_number():
    pages = parse_pdf(make_pdf())
    assert pages == [(2, "At least two reviewers.")]
    chunks = split_pages(pages, 10, 2)
    assert all(chunk.page == 2 for chunk in chunks)
    assert [chunk.offset for chunk in chunks] == [0, 8, 16]


def test_upload_chat_and_delete(tmp_path):
    embedder = Mock()
    embedder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    generator = Mock()
    generator.answer.return_value = Draft(
        sufficient=True, answer="At least two.", citation_ids=[1]
    )
    with TestClient(create_app(Settings(data_dir=tmp_path), embedder, generator)) as client:
        assert client.get("/health").status_code == 200
        files = {"file": ("demo.pdf", make_pdf(), "application/pdf")}
        upload = client.post("/api/documents/upload", files=files)
        assert upload.status_code == 201, upload.text
        assert client.post("/api/documents/upload", files=files).status_code == 409
        payload = {"question": "How many reviewers?", "session_id": "demo"}
        answer = client.post("/api/chat", json=payload)
        assert answer.status_code == 200, answer.text
        assert answer.json()["sources"][0]["page"] == 2
        assert len(client.get("/api/chat/history/demo").json()) == 1
        assert client.post("/api/chat", json={**payload, "question": ""}).status_code == 422
        document_id = upload.json()["id"]
        assert client.delete(f"/api/documents/{document_id}").status_code == 204
        assert client.get("/api/documents").json() == []
        assert client.post("/api/chat", json=payload).json()["sources"] == []


def test_invalid_citation_is_discarded(tmp_path):
    embedder = Mock()
    embedder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    generator = Mock()
    generator.answer.return_value = Draft(
        sufficient=True, answer="Unsupported claim.", citation_ids=[99]
    )
    with TestClient(create_app(Settings(data_dir=tmp_path), embedder, generator)) as client:
        files = {"file": ("demo.pdf", make_pdf(), "application/pdf")}
        assert client.post("/api/documents/upload", files=files).status_code == 201
        result = client.post("/api/chat", json={"question": "Reviewers?", "session_id": "demo"})
        assert result.status_code == 200
        assert result.json()["sources"] == []
        assert "证据不足" in result.json()["answer"]
