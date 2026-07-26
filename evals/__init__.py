from typing import Any

__all__ = ["EvalRunnerConfig", "run_eval_suite"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from evals.runner import EvalRunnerConfig, run_eval_suite

    return {
        "EvalRunnerConfig": EvalRunnerConfig,
        "run_eval_suite": run_eval_suite,
    }[name]
