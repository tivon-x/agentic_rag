from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def test_frozen_route_dataset_has_balanced_four_way_labels():
    path = Path("evals/datasets/m4_1_route_v1.json")
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) == 48
    assert Counter(case["expected_route"] for case in cases) == {"direct": 12, "fixed": 12, "adaptive": 12, "refuse": 12}
    assert all("authoring_source" in case and "required_facts" in case for case in cases)
