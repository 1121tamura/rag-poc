import re
from datetime import datetime
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode

_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$")


def load_markdown(file_path: Path) -> list[TextNode]:
    """Markdownファイルを読み込み、見出し単位のチャンク（TextNode）に分割する"""
    text = file_path.read_text(encoding="utf-8")
    document = Document(text=text)
    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents([document])


def extract_chapter_title(text: str) -> str | None:
    """チャンク本文の先頭行が見出しなら、そのタイトル文字列を返す"""
    first_line = text.splitlines()[0] if text else ""
    match = _HEADING_PATTERN.match(first_line)
    return match.group(1).strip() if match else None


def attach_metadata(
    nodes: list[TextNode],
    *,
    document_type: str,
    file_path: Path,
    file_format: str,
    updated_at: datetime,
) -> list[TextNode]:
    """チャンクにメタデータ（document_type/file_path/chapter_title/file_format/updated_at）を付与する"""
    for node in nodes:
        node.metadata["document_type"] = document_type
        node.metadata["file_path"] = str(file_path)
        node.metadata["file_format"] = file_format
        node.metadata["updated_at"] = updated_at.isoformat()
        node.metadata["chapter_title"] = extract_chapter_title(node.text)
    return nodes
