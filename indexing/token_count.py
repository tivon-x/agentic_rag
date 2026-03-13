from __future__ import annotations

import re

import tiktoken


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[A-Za-z0-9_]+|[^\w\s]",
    re.UNICODE,
)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    try:
        return len(_ENCODING.encode(stripped))
    except Exception:
        if _CJK_RE.search(stripped):
            return len(_TOKEN_RE.findall(stripped))
        return len(stripped.split())
