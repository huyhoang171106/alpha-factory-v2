import os

filepath = r'd:\alpha-factory-private\run_async_pipeline.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_simulator = False
for line in lines:
    if "async def worker_simulator" in line:
        in_simulator = True
    
    if in_simulator and "while self.is_running:" in line:
        new_lines.append(line)
        new_lines.append("            cand = None\n")
        in_simulator = False # Only add it to the first while loop in simulator
        continue
    
    if "if 'cand' in locals():" in line:
        new_lines.append(line.replace("if 'cand' in locals():", "if cand is not None:"))
        continue
        
    new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Done")
