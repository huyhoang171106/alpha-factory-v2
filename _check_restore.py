# Fix the truncation in community_harvester.py
import sys, shutil, os

backup = r'D:\alpha-factory-private\community_harvester.py.bak'
src   = r'D:\alpha-factory-private\community_harvester.py'

# Restore from backup if it exists
if os.path.exists(backup):
    shutil.copy(backup, src)
    print('Restored from backup')
else:
    print('No backup found — checking current file length...')

with open(src, 'rb') as f:
    raw = f.read()
print(f'File size: {len(raw)} bytes')

# Show a safe snippet near the end
idx = raw.find(b'mutant = original[:i]')
if idx >= 0:
    print('mutant = original[:i] found at byte', idx)
    print('Context:')
    print(repr(raw[idx-100:idx+200]))