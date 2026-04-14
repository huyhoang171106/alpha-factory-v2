import subprocess
import os
import sys

def run_tests():
    root = os.getcwd()
    venv_python = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = "python" # fallback
    
    cmd = [venv_python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    print(f"Running: {' '.join(cmd)}")
    
    with open("test_results.log", "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, text=True)
        try:
            proc.wait(timeout=300) # 5 minutes
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Timeout!")
            return 1
    
    return proc.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
