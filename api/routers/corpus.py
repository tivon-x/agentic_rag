from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_settings
from api.models.corpus import CorpusProfile
from core.corpus_profile import load_corpus_profile, normalize_corpus_profile, save_corpus_profile
from core.settings import AppSettings


router = APIRouter(prefix="/corpus-profile", tags=["corpus-profile"])


@router.get("", response_model=CorpusProfile)
async def get_corpus_profile(settings: AppSettings = Depends(get_settings)) -> CorpusProfile:
    return CorpusProfile.model_validate(load_corpus_profile(settings.index_dir))


@router.put("", response_model=CorpusProfile)
async def update_corpus_profile(
    profile: CorpusProfile,
    settings: AppSettings = Depends(get_settings),
) -> CorpusProfile:
    normalized = normalize_corpus_profile(profile.model_dump())
    save_corpus_profile(
        settings.index_dir,
        name=normalized["name"],
        summary=normalized["summary"],
        coverage=normalized["coverage"],
        non_coverage=normalized["non_coverage"],
        usage_notes=normalized["usage_notes"],
        source_examples=normalized["source_examples"],
        recommended_questions=normalized["recommended_questions"],
        forbidden_questions=normalized["forbidden_questions"],
        domain_keywords=normalized["domain_keywords"],
        preferred_answer_style=normalized["preferred_answer_style"],
        primary_entities=normalized["primary_entities"],
    )
    return CorpusProfile.model_validate(load_corpus_profile(settings.index_dir))
