from __future__ import annotations

from agent.adaptive_graph import _route
from agent.edges import route_after_adaptive_decision


def test_adaptive_route_edges_keep_direct_and_refuse_out_of_retrieval():
    assert route_after_adaptive_decision({"strategy": "direct"}) == "adaptive_direct"
    assert route_after_adaptive_decision({"strategy": "refuse"}) == "adaptive_refuse"
    assert route_after_adaptive_decision({"strategy": "fact"}) == "adaptive_fact"


def test_direct_and_live_external_requests_do_not_depend_on_model_routing():
    assert _route("你好") == "direct"
    assert _route("上海今天的天气怎样？") == "refuse"
