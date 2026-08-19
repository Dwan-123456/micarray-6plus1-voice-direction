VALID_BEAM_DIRECTIONS = tuple("0123456789AB")


def led_command(enabled: bool) -> bytes:
    return b"E" if enabled else b"e"


def beam_direction_command(direction: str) -> bytes:
    normalized = direction.strip().upper()
    if normalized not in VALID_BEAM_DIRECTIONS:
        raise ValueError("direction 必须是 0..9、A 或 B")
    return normalized.encode("ascii")


def threshold_command(increase: bool) -> bytes:
    return b"T" if increase else b"t"


def restore_defaults_command() -> bytes:
    return b"R"
