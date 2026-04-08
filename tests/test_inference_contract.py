import io
from contextlib import redirect_stdout

import inference


def test_log_format_is_exact() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        inference.log_start(task="task", env="env", model="model")
        inference.log_step(step=1, action='{"action_type":"submit"}', reward=1.0, done=True, error=None)
        inference.log_end(success=True, steps=1, score=1.0, rewards=[1.0])
    lines = buf.getvalue().strip().splitlines()
    assert lines[0] == "[START] task=task env=env model=model"
    assert lines[1] == '[STEP] step=1 action={"action_type":"submit"} reward=1.0 done=True error=None'
    assert lines[2] == "[END] success=True steps=1 score=1.0 rewards=[1.0]"


def test_task_order_and_fallback_contract() -> None:
    assert inference.TASK_ORDER == [
        "easy_verified_access_with_injection",
        "medium_unverified_erasure_multi_account",
        "hard_guardian_minor_legal_hold_fraud",
    ]
    assert inference.fallback_policy("easy_verified_access_with_injection", 1)["action_type"] == "inspect_case"
    assert inference.fallback_policy("hard_guardian_minor_legal_hold_fraud", 20)["action_type"] == "submit"
