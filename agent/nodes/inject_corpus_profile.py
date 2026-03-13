from agent.states import GraphState


def inject_corpus_profile(
    corpus_profile: str,
    corpus_profile_data: dict | None = None,
):
    def _inject(_: GraphState):
        return {
            "corpusProfile": corpus_profile,
            "corpusProfileData": dict(corpus_profile_data or {}),
        }

    return _inject
