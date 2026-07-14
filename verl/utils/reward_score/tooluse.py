# Reward scorer for the ToolCalling (tooluse) dataset (context-distillation port).
#
# Mirrors SDFT/asymrl's correctness_reward_func: the response is correct iff
#   (a) the multiset of extracted action NAMES equals the gold multiset, AND
#   (b) the merged Action_Input dict equals the gold merged dict.
# 1.0 / 0.0.
#
# ``ground_truth`` here is the JSON-serialised golden_answer action list, e.g.
#   '[{"Action": "sendHttpRequest", "Action_Input": "{\\"method\\": \\"POST\\", ...}"}]'
# (the tooluse dataset builder json.dumps() it).

import json
import re
from collections import Counter


def extract_action(text: str) -> list[str]:
    """Action names from a model response (``Action: <name>``)."""
    return re.findall(r"Action:\s*(\w+)", text)


def extract_action_inputs(text: str) -> dict:
    """Merged ``Action Input: {...}`` JSON objects from a model response.

    NOTE — deliberate divergence from SDFT/asymrl (issue #27). asymrl uses a
    NON-greedy regex ``Action Input:\\s*({.*?})`` which stops at the first ``}``
    and so fails to parse any ``Action_Input`` containing a NESTED object (~5% of
    tooluse gold answers), scoring even a perfect completion 0 — a silently dead
    reward channel (cf. #26). We instead scan for each ``Action Input:`` and take
    the BALANCED ``{...}`` block. Strict superset of asymrl on flat inputs; lifts
    gold-demo self-score 95%% -> 99.7%%."""
    combined: dict = {}
    for m in re.finditer(r"Action Input:\s*", text):
        start = text.find("{", m.end())
        if start < 0:
            continue
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        combined.update(json.loads(text[start : j + 1]))
                    except json.JSONDecodeError:
                        pass
                    break
    return combined


def _merge(list_of_dicts) -> dict:
    combined: dict = {}
    for d in list_of_dicts:
        if d:
            combined.update(d)
    return combined


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    """1.0 iff action-name multiset AND merged Action_Input dict both match gold."""
    gold = ground_truth
    if isinstance(gold, str):
        gold = json.loads(gold)

    gt_actions = [item["Action"] for item in gold]
    gt_action_inputs = _merge(json.loads(item["Action_Input"]) for item in gold)

    actions = extract_action(solution_str)
    action_inputs = extract_action_inputs(solution_str)

    actions_ok = Counter(actions) == Counter(gt_actions)
    inputs_ok = action_inputs == gt_action_inputs
    return 1.0 if actions_ok and inputs_ok else 0.0
