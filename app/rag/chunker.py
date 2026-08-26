"""文档加载与文本分块。

P1：实现「段落聚合 + 长度/重叠切分」的朴素分块器，零依赖、可读、可测。
进阶可换 langchain-text-splitters 或按语义分块，README 有说明。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def _units(text: str) -> list[str]:
    """按行（段落）切分，过滤空行。"""
    return [line.strip() for line in text.splitlines() if line.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """把长文本切成带重叠的块。

    策略：按段落顺序累积，超过 chunk_size 就封块，并保留 overlap 长度的尾部
    作为下一块的开头，避免关键信息被拦腰截断。
    """
    units = _units(text)
    if not units:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append(Chunk(text="\n".join(buf), source=source, chunk_index=len(chunks)))
        tail = "\n".join(buf)[-overlap:] if overlap > 0 else ""
        buf = [tail] if tail.strip() else []
        buf_len = len(tail)

    for unit in units:
        # 单个段落超过 chunk_size：按固定窗口硬切
        while len(unit) > chunk_size:
            if buf:
                flush()
            chunks.append(Chunk(text=unit[:chunk_size], source=source, chunk_index=len(chunks)))
            unit = unit[chunk_size - overlap:]

        if buf_len + len(unit) + 1 > chunk_size:
            flush()
        buf.append(unit)
        buf_len += len(unit) + 1

    flush()
    return chunks


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path: Path) -> str:
    """按扩展名加载文档：PDF -> pypdf，docx -> python-docx，其余按纯文本。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        import docx

        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    return path.read_text(encoding="utf-8")
