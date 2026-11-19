from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class Chunk:
    page: int
    offset: int
    text: str


def parse_pdf(data: bytes) -> list[tuple[int, str]]:
    if not data.startswith(b"%PDF-"):
        raise ValueError("文件内容缺少 PDF 标识。")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("请先解除 PDF 密码保护。")
            if pdf.page_count > 500:
                raise ValueError("PDF 不得超过 500 页。")
            pages = []
            chars = 0
            for page in pdf:
                text = page.get_text(sort=True).strip()
                chars += len(text)
                if chars > 2_000_000:
                    raise ValueError("PDF 提取文本超过限制。")
                if text:
                    pages.append((page.number + 1, text))
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("PDF 无法读取，文件可能已损坏。") from exc
    if not pages:
        raise ValueError("PDF 无可提取文本。扫描件需要先进行 OCR。")
    return pages


def split_pages(
    pages: list[tuple[int, str]], size: int, overlap: int
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("要求 0 ≤ chunk_overlap < chunk_size。")
    chunks = []
    for page, text in pages:
        for start in range(0, len(text), size - overlap):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append(Chunk(page, start, piece))
            if start + size >= len(text):
                break
    return chunks
