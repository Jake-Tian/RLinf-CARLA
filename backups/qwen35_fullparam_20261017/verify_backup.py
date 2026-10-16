"""Verify this source backup without importing training dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text())
for rel, expected in manifest["files"].items():
    path = root / rel
    assert path.is_file(), rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], rel
    if path.suffix == ".py":
        ast.parse(path.read_text(), filename=rel)
repo = root.parents[1]
assert (repo / ".git").exists(), "Run from the full backup repository"
with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    for rel in sorted(set(manifest["framework_async_changes"] + manifest["framework_colocate_changes"])):
        result = subprocess.run(["git", "show", manifest["public_base_commit"] + ":" + rel], cwd=repo, capture_output=True)
        if result.returncode == 0:
            target = temp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.stdout)
    subprocess.run(["git", "apply", "--check", str(root / "patches/rlinf_async.patch")], cwd=temp, check=True)
    subprocess.run(["git", "apply", str(root / "patches/rlinf_async.patch")], cwd=temp, check=True)
    for rel in manifest["framework_async_changes"]:
        assert (temp / rel).read_bytes() == (root / "source/async" / rel).read_bytes(), rel
    subprocess.run(["git", "apply", "--check", str(root / "patches/colocate_on_async.patch")], cwd=temp, check=True)
    subprocess.run(["git", "apply", str(root / "patches/colocate_on_async.patch")], cwd=temp, check=True)
    for rel in manifest["framework_colocate_changes"]:
        assert (temp / rel).read_bytes() == (root / "source/colocate" / rel).read_bytes(), rel
print("PASS", len(manifest["files"]), "files: hashes, Python syntax, async reconstruction, colocate patch applicability")
