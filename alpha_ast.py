"""
alpha_ast.py - Pseudo-AST for Alpha Expressions
Allows safe Crossover of Alpha Strings without native python `ast` limits.
Provides functions to extract semantic node blocks from Brain alphas and swap them.
"""
import random
import re

try:
    from validator import normalize_expression_aliases
except Exception:
    def normalize_expression_aliases(expr: str) -> str:
        return expr

def extract_nodes(expr: str) -> list:
    """Find all bracketed (nested) logic parts in the alpha expression."""
    nodes = []
    depth = 0
    start = -1
    for i, char in enumerate(expr):
        if char == '(':
            if depth == 0: start = i
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0 and start != -1:
                # Capture the full block including its parent function name if possible
                # e.g., for "rank(x)", we want to capture the whole thing or just "(x)"
                # This finds the closest preceding word boundary
                func_start = start
                while func_start > 0 and (expr[func_start-1].isalnum() or expr[func_start-1] == '_'):
                    func_start -= 1
                nodes.append((func_start, i+1, expr[func_start:i+1]))
    return nodes

def tree_crossover(parent1: str, parent2: str) -> str:
    """Take a random tree node from parent2 and inject it into parent1's node."""
    nodes1 = extract_nodes(parent1)
    nodes2 = extract_nodes(parent2)
    
    # If no outer matching parentheses logic exist, fallback
    if not nodes1 or not nodes2:
        return parent1
    
    n1 = random.choice(nodes1)
    n2 = random.choice(nodes2)
    
    child = parent1[:n1[0]] + n2[2] + parent1[n1[1]:]
    return child


def canonicalize_expression(expr: str) -> str:
    """
    Normalize expression for stable comparison:
    - alias normalization
    - lowercase
    - remove all whitespace
    """
    normalized = normalize_expression_aliases((expr or "").strip()).lower()
    return re.sub(r"\s+", "", normalized)


def parameter_agnostic_signature(expr: str) -> str:
    """
    Structure signature that ignores numeric parameter values.
    Example: ts_delta(close,5) and ts_delta(close,10) map to same family signature.
    """
    canonical = canonicalize_expression(expr)
    # Replace integer/float constants with generic marker.
    return re.sub(r"\b\d+(\.\d+)?\b", "N", canonical)


def token_set(expr: str, strip_numbers: bool = False) -> set[str]:
    """Tokenize expression into a set for fast overlap checks."""
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?"
    tokens = set(re.findall(pattern, canonicalize_expression(expr)))
    if strip_numbers:
        tokens = {t for t in tokens if not re.fullmatch(r"\d+(?:\.\d+)?", t)}
    return tokens


def operator_set(expr: str) -> set[str]:
    """Return operator/function names used in expression."""
    canonical = canonicalize_expression(expr)
    return set(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\(", canonical))

if __name__ == "__main__":
    p1 = "rank(ts_delta(close, 5))"
    p2 = "group_neutralize(ts_mean(volume, 20), sector)"
    print("P1:", p1)
    print("P2:", p2)
    print("Crossover:", tree_crossover(p1, p2))
