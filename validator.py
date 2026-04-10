"""
validator.py — Alpha Expression Syntax Validator
Checks expression validity before sending to WQ Brain API.
Reference: worldquant-miner/generation_two/core/fast_expr_ast.py
"""

import re
from typing import Tuple

# Valid WQ Brain FASTEXPR operators (Source: operatorRAW.json catalog)
VALID_OPERATORS = {
    # Time-series operators
    "ts_mean", "ts_sum", "ts_std_dev", "ts_min", "ts_max",
    "ts_rank", "ts_zscore", "ts_delta", "ts_delay",
    "ts_corr", "ts_covariance", "ts_regression",
    "ts_arg_max", "ts_arg_min", "ts_product",
    "ts_decay_linear", "ts_decay_exp_window",
    "ts_skewness", "ts_kurtosis", "ts_step",
    "ts_backfill", "ts_quantile", "ts_scale",
    "ts_av_diff", "ts_count_nans",
    "ts_target_tvr_decay", "ts_target_tvr_delta_limit",
    # Rank / normalize / arithmetic
    "rank", "zscore", "sigmoid", "sign", "signed_power",
    "scale", "power", "log", "abs", "min", "max",
    "sqrt", "floor", "inverse", "reverse", "normalize",
    "quantile", "scale_down", "winsorize",
    # Group operators
    "group_rank", "group_zscore", "group_neutralize",
    "group_mean", "group_median", "group_sum",
    "group_max", "group_min", "group_normalize",
    "group_scale", "group_count", "group_percentage",
    # Vector operators
    "vec_sum", "vec_avg", "vec_max", "vec_min",
    "vector_neut",
    # Transformational / Special
    "pasteurize", "trade_when", "Ts_Rank",
    "last_diff_value", "days_from_last_change",
    "bucket", "densify", "hump", "jump_decay",
    "kth_element", "if_else", "is_nan", "to_nan",
    # Logical (used as operators in ternary expressions)
    "add", "subtract", "multiply", "divide",
}

# Valid data fields
VALID_FIELDS = {
    # Price data
    "open", "high", "low", "close", "volume", "vwap",
    "returns", "cap", "sharesout", "adv20", "adv60", "adv120",
    # Fundamental & Alt data
    "sales", "eps", "book_value", "ebitda", "debt",
    "income", "dividends", "assets", "equity",
    "operating_margin", "return_equity", "sentiment", 
    "short_interest_ratio", "analyst_estimates",
    # Group labels
    "market", "sector", "industry", "subindustry",
}

MAX_EXPR_LENGTH = 1024
RESERVED_IDENTIFIERS = {"LAG", "RETTYPE", "true", "false", "nan", "none", "e"}

# Common aliases emitted by external generators/LLMs.
FIELD_ALIASES = {
    "ret": "returns",
    "adv50": "adv60",
}
OPERATOR_ALIASES = {
    "delay": "ts_delay",
    "delta": "ts_delta",
    "ts_std": "ts_std_dev",
    "ts_cov": "ts_covariance",
}


def normalize_expression_aliases(expr: str) -> str:
    """
    Normalize common field/operator aliases into canonical FASTEXPR tokens.
    This reduces avoidable rejects before hitting WQ API.
    """
    normalized = expr
    for old, new in {**FIELD_ALIASES, **OPERATOR_ALIASES}.items():
        normalized = re.sub(rf"\b{re.escape(old)}\b", new, normalized)
    return normalized


def validate_expression(expr: str) -> Tuple[bool, str]:
    """
    Validate an alpha expression for WQ Brain.
    
    Returns: (is_valid, error_message)
    """
    expr = normalize_expression_aliases(expr.strip())

    # 1. Empty check
    if not expr:
        return False, "Empty expression"

    # 2. Length check
    if len(expr) > MAX_EXPR_LENGTH:
        return False, f"Expression too long ({len(expr)} > {MAX_EXPR_LENGTH})"

    # 3. Balanced parentheses
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False, "Unbalanced parentheses (extra closing)"
    if depth != 0:
        return False, f"Unbalanced parentheses (unclosed: {depth})"

    # 4. Check for obviously invalid patterns
    if "()" in expr:
        return False, "Empty parentheses '()' found"

    if ",," in expr:
        return False, "Double comma ',,' found"

    if expr.count('"') % 2 != 0:
        return False, "Unmatched quotes"

    # 5. Check operators exist in the expression
    # Extract function calls: word followed by (
    func_calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', expr)
    for func in func_calls:
        if func not in VALID_OPERATORS and func.lower() not in VALID_OPERATORS:
            # Check if it's a known group (sector, etc.) — not a function
            if func not in VALID_FIELDS:
                return False, f"Unknown operator: '{func}'"

    # 5.5 Check unknown identifiers (non-function tokens like fields/labels)
    # Prevents silent leaks such as "ret" or typo fields passing local validation.
    func_set = set(func_calls)
    # Ignore quoted strings while token scanning (e.g., driver='cauchy').
    expr_no_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "", expr)
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr_no_strings)
    named_arg_keys = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', expr_no_strings))
    for token in identifiers:
        if token in func_set:
            continue
        if token in named_arg_keys:
            continue
        if token in VALID_FIELDS or token in VALID_OPERATORS:
            continue
        if token in RESERVED_IDENTIFIERS or token.upper() in RESERVED_IDENTIFIERS:
            continue
        return False, f"Unknown identifier: '{token}'"

    # 6. Check for division by zero patterns
    if "/0)" in expr or "/0 " in expr or "/0," in expr:
        return False, "Possible division by zero"

    return True, "OK"


def validate_batch(expressions: list[str]) -> list[str]:
    """
    Validate a batch of expressions.
    Returns only valid expressions.
    """
    valid = []
    for expr in expressions:
        is_valid, msg = validate_expression(expr)
        if is_valid:
            valid.append(normalize_expression_aliases(expr))
    return valid


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    tests = [
        "rank(ts_delta(close, 5))",           # Valid
        "-rank(ts_corr(open, volume, 10))",    # Valid
        "rank(ts_delta(close, 5)",             # Invalid: unbalanced
        "foo_bar(close, 5)",                   # Invalid: unknown op
        "",                                     # Invalid: empty
        "rank(close / 0)",                      # Risky: div by zero
        "a" * 1100,                             # Invalid: too long
    ]

    for t in tests:
        ok, msg = validate_expression(t)
        status = "✅" if ok else "❌"
        print(f"{status} {msg:<40} | {t[:60]}")
