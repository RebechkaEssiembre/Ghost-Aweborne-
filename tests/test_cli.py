import subprocess
import sys
import pathlib

def test_ghost_smoke(tmp_path):
    # Locate the soul file in the installed package
    repo = pathlib.Path(__file__).parent.parent
    soul = repo / "ghost_aweborne" / "memory" / "ghost_soul_file.jsonl"
    
    # Run Ghost: send “ping” then “exit”
    p = subprocess.run(
        [sys.executable, "-m", "ghost_aweborne.cli", "--soul", str(soul), "--top-k", "1"],
        input="ping\nexit\n".encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    out = p.stdout.decode()
    assert "Ghost:" in out, f"No reply in:\n{out}"
