def resolve_terminal_outcome(
    previous_success,
    previous_collision,
    target,
    collision,
):
    if previous_collision or collision:
        return False, True
    if previous_success or target:
        return True, False
    return False, False
