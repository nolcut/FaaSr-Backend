from pathlib import Path
import subprocess
import os

def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(f"Command: {' '.join(cmd)}")
    print(f"Return code: {p.returncode}")
    print(f"Output:\n{out}")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out

def main():
    
    repo_root = Path(__file__).resolve().parents[3]  # tests -> faasr_ai -> src -> repo
    register_script = repo_root / "register_workflow.sh"
    workflow_file = "workflows/tutorial.json"
    
    print(f"repo_root: {repo_root}")
    print(f"register_script exists: {register_script.exists()}")
    print(f"workflow_file exists: {(repo_root / workflow_file).exists()}")
    
    out = run_cmd(["bash", str(register_script), "--workflow-file", workflow_file, "-c"], cwd=repo_root)
    
    print(out)
    
    invoke_script = repo_root / "invoke_workflow.sh"
    
    out = run_cmd(["bash", str(invoke_script), "--workflow-file", workflow_file], cwd=repo_root)
    
    print(out)

if __name__ == "__main__":
    main()