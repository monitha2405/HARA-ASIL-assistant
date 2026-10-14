"""Deterministic ASIL rule (ISO 26262-3 style table, S+E+C sum form).
Verify against your licensed copy of the standard before real use."""
def asil(s: int, e: int, c: int) -> str:
    if s == 0 or e == 0 or c == 0:
        return "QM"
    return {10: "D", 9: "C", 8: "B", 7: "A"}.get(s + e + c, "QM")
ORDER = ["QM", "A", "B", "C", "D"]
