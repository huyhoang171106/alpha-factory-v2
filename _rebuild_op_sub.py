"""
Rebuild apply_operator_substitution cleanly.
Reads the damaged file, replaces the corrupted method with the correct version.
"""
import sys

F = r'D:\alpha-factory-private\community_harvester.py'

with open(F, 'rb') as f:
    raw = f.read()

CORRECT_METHOD = (
    b'    def apply_operator_substitution(self, alpha_expr: str) -> list[str]:\n'
    b'        """\n'
    b'        For each outermost operator in the alpha that belongs to a semantic\n'
    b'        equivalence group, replace it with its equivalent counterpart,\n'
    b'        preserving all arguments intact.  Each substitution is built from\n'
    b'        the original string (never from a previously-mutated result) to\n'
    b'        prevent cascading cross-contamination between rules.\n'
    b'\n'
    b'        Returns a list of mutated expressions (skips if nothing changed or\n'
    b'        the result has unbalanced parentheses).\n'
    b'        """\n'
    b'        original = alpha_expr.strip()\n'
    b'        mutants: list[str] = []\n'
    b'\n'
    b'        for old_op, new_op in self.OPERATOR_EQUIVALENCE_GROUPS:\n'
    b'            i = 0\n'
    b'            while i < len(original):\n'
    b'                if original[i:i + len(old_op)] == old_op:\n'
    b'                    if i > 0 and original[i - 1].isalnum():\n'
    b'                        i += 1\n'
    b'                        continue\n'
    b'                    # Depth check: scan chars to LEFT of position i\n'
    b'                    depth = 0\n'
    b'                    for j in range(i):\n'
    b'                        if original[j] == 40:  depth += 1\n'
    b'                        elif original[j] == 41: depth -= 1\n'
    b'                    if depth != 0:\n'
    b'                        i += 1\n'
    b'                        continue\n'
    b'                    # Unified depth scan for arguments\n'
    b'                    rest = original[i + len(old_op):]\n'
    b'                    depth = 1\n'
    b'                    arg_end = len(rest)\n'
    b'                    for k, ch in enumerate(rest):\n'
    b'                        if ch == 40:  depth += 1\n'
    b'                        elif ch == 41:\n'
    b'                            depth -= 1\n'
    b'                            if depth == 0:\n'
    b'                                arg_end = k\n'
    b'                                break\n'
    b'                    args = rest[:arg_end]\n'
    b'                    mutant = original[:i] + new_op + args\n'
    b'                    if (mutant != original and\n'
    b'                            mutant.count(40) == mutant.count(41)):\n'
    b'                        mutants.append(mutant)\n'
    b'                    i += len(old_op)\n'
    b'                    continue\n'
    b'                i += 1\n'
    b'        return mutants\n'
)

# Locate method boundaries
method_start = raw.find(b'def apply_operator_substitution')
method_end   = raw.find(b'def decompose_and_recombine')
print(f'Method at bytes {method_start}..{method_end}  (len={method_end-method_start})')

if method_start == -1:
    print('ERROR: method not found'); sys.exit(1)

before = raw[:method_start]
after  = raw[method_end:]
new_raw = before + CORRECT_METHOD + after

with open(F, 'wb') as f:
    f.write(new_raw)
print(f'Done. New file: {len(new_raw)} bytes')
