with open(r'D:\alpha-factory-private\community_harvester.py', 'rb') as f:
    raw = f.read()

start = raw.find(b'if rest.startswith')
end = raw.find(b'mutant = original[:i]', start)
old_bytes = raw[start:end]

# Build new block with CORRECT indentation (20 spaces = 5 levels)
new_block = (
    b'                    rest = original[i + len(old_op):]\n\n'
    b'                    # Unified depth scan: start at depth 1 (inside the call)\n'
    b'                    # and collect chars until we exit back to depth 0.\n'
    b'                    # Works for: rank(close), rank(ts_zscore(v)), foo(bar(x),baz(y))\n'
    b'                    depth = 1\n'
    b'                    arg_end = len(rest)\n'
    b'                    for k, ch in enumerate(rest):\n'
    b'                        if ch == 40:  # ord("(")\n'
    b'                            depth += 1\n'
    b'                        elif ch == 41:  # ord(")")\n'
    b'                            depth -= 1\n'
    b'                            if depth == 0:\n'
    b'                                arg_end = k\n'
    b'                                break\n'
    b'                    args = rest[:arg_end]\n\n'
    b'                    '
)

new_raw = raw[:start] + new_block + raw[end:]
with open(r'D:\alpha-factory-private\community_harvester.py', 'wb') as f:
    f.write(new_raw)
print('Patched successfully')
print(f'Replaced {len(old_bytes)} bytes with {len(new_block)} bytes')
