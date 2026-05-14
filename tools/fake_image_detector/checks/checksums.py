def validate_mrz_digit(value: str) -> bool:
    """ICAO Doc 9303 weighted 7/3/1 mod-10 check digit."""
    weights = [7, 3, 1]
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    try:
        total = sum(
            chars.index(c) * weights[i % 3]
            for i, c in enumerate(value[:-1])
            if c != "<"
        )
    except ValueError:
        return False
    try:
        return (total % 10) == int(value[-1])
    except (ValueError, IndexError):
        return False


def validate_luhn(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not digits:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validate_iban(value: str) -> bool:
    iban = value.replace(" ", "").upper()
    if len(iban) < 5:
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
        return int(numeric) % 97 == 1
    except ValueError:
        return False
