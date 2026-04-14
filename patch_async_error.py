import sys
import os

filepath = r'd:\alpha-factory-private\run_async_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # 1. Add cand = None after while
    if "while self.is_running:" in line and "worker_simulator" in "".join(lines[lines.index(line)-10:lines.index(line)]):
        new_lines.append(line)
        new_lines.append("            cand = None\n")
        continue
    
    # 2. Fix finally block
    if "if 'cand' in locals():" in line:
        new_lines.append(line.replace("if 'cand' in locals():", "if cand is not None:"))
        continue
        
    new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Done")
