# tools/calculator.py

def math_eval(expression: str) -> str:
    """Evaluates a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"