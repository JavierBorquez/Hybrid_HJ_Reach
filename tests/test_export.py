"""Curated exports contain only allowed public files and verifiable hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("curator", Path(__file__).parents[1]/"tools/curate.py")
curator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curator)


def test_export_excludes_outputs_and_private_artifacts(tmp_path):
    destination = tmp_path/"public"
    curator.curate(destination)
    manifest = json.loads((destination/"EXPORT_MANIFEST.json").read_text())
    assert "src/hybrid_reach/solver.py" in manifest["sha256"]
    assert "tutorials/falling_piece.py" in manifest["sha256"]
    assert "docs/images/falling_piece_reach_avoid_policy.gif" in manifest["sha256"]
    assert "AGENTS.md" in manifest["sha256"]
    for name, digest in manifest["sha256"].items():
        path = Path(name)
        assert path.parts[0] not in ("outputs", "references", ".git", ".venv")
        assert path.suffix not in (".npy", ".npz", ".mat", ".pdf", ".log", ".xml")
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        curator.curate(destination)


def test_export_rejects_destination_inside_development_tree():
    with pytest.raises(ValueError):
        curator.curate(Path(__file__).parents[1]/"nested-export")
