"""Action-type feature engineering."""

ACTION_CATEGORY_MAP = {
    "Jump Shot": "jump_shot",
    "Pullup Jump shot": "pullup",
    "Step Back Jump shot": "stepback",
    "Fadeaway Jump Shot": "fadeaway",
    "Running Jump Shot": "running_jump",
    "Floating Jump shot": "floater",
    "Turnaround Jump Shot": "turnaround",
    "Hook Shot": "hook",
    "Running Hook Shot": "hook",
    "Driving Hook Shot": "hook",
    "Turnaround Hook Shot": "hook",
}

LAYUP_KEYWORDS = ("Layup", "Finger Roll", "Reverse Layup", "Cutting")
DUNK_KEYWORDS = ("Dunk", "Alley Oop", "Tip Dunk")
FLOATER_KEYWORDS = ("Floating", "Floater")


def categorize_action(action: str) -> str:
    if not isinstance(action, str):
        return "other"

    if any(k in action for k in DUNK_KEYWORDS):
        return "dunk"
    if any(k in action for k in LAYUP_KEYWORDS):
        return "layup"
    if any(k in action for k in FLOATER_KEYWORDS):
        return "floater"

    if action in ACTION_CATEGORY_MAP:
        return ACTION_CATEGORY_MAP[action]

    if "Pullup" in action or "Pull-Up" in action:
        return "pullup"
    if "Step Back" in action or "Stepback" in action:
        return "stepback"
    if "Fadeaway" in action:
        return "fadeaway"
    if "Hook" in action:
        return "hook"
    if "Jump Shot" in action or "Jumper" in action:
        return "jump_shot"

    return "other"
