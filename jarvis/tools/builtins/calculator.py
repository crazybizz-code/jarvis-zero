from __future__ import annotations

import ast
import operator

from jarvis.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# Safe expression evaluator — no eval(), no exec()
#
# We parse the expression into an AST then walk only the node types we
# explicitly allow. Any other node (function calls, imports, attribute
# access, comprehensions, ...) raises ValueError before anything executes.
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type, callable] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

_UNARY_OPS: dict[type, callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPONENT = 1_000  # guard against 9999**9999 hanging the process


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported literal type: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent {right} exceeds limit of {_MAX_EXPONENT}.")
        if op_type is ast.Div and right == 0:
            raise ZeroDivisionError("Division by zero.")
        if op_type is ast.FloorDiv and right == 0:
            raise ZeroDivisionError("Division by zero.")
        return _BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only arithmetic with numbers is allowed."
    )


def safe_eval(expression: str) -> int | float:
    """Parse and evaluate a pure arithmetic expression. Raises on anything unsafe."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
    return _eval_node(tree.body)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class CalculatorTool(BaseTool):
    """Evaluates arithmetic expressions safely, without using eval()."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluates a mathematical expression and returns the result. "
            "Supports +, -, *, /, //, %, ** and parentheses. "
            "Use this for any arithmetic the user asks you to compute."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate, e.g. '(3 + 4) * 2'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs) -> ToolResult:
        expression = kwargs.get("expression", "")
        try:
            result = safe_eval(expression)
            # Return integers without a decimal point; floats with full precision
            formatted = str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
            return ToolResult(content=f"{expression} = {formatted}")
        except (ValueError, ZeroDivisionError) as exc:
            return ToolResult(content=str(exc), error=True)
