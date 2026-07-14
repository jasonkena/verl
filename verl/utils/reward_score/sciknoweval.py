# Reward scorer for the SciKnowEval science-MCQ dataset (context-distillation port).
#
# Mirrors SDFT/asymrl's science_reward_func: extract the letter between
# <answer>...</answer> and exact-match it against the gold letter. 1.0 / 0.0.


def extract_xml_answer(text: str) -> str:
    """Letter between the last ``<answer>`` and the following ``</answer>``
    (matches SDFT eval_science.py / asymrl extract_xml_answer)."""
    a = text.split("<answer>")[-1]
    a = a.split("</answer>")[0]
    return a.strip()


def compute_score(solution_str, ground_truth, **kwargs) -> float:
    """1.0 if the extracted <answer> letter matches the gold letter, else 0.0."""
    return 1.0 if extract_xml_answer(solution_str) == str(ground_truth).strip() else 0.0
