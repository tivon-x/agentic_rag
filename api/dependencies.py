from __future__ import annotations

from fastapi import Request

from core.settings import AppSettings


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings
