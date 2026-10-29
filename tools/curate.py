"""Create a fresh, allowlisted source export; never copy private Git history/data."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def curate(destination):
    source = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Export destination must be outside the development export directory")
    if destination.exists():
        raise FileExistsError("Choose a new export destination")
    paths = []
    for directory in ("src", "tests", "tutorials", "examples", "docs", "tools"):
        paths.extend(p for p in (source/directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".md", ".png") and not p.is_symlink())
    paths.extend(source/name for name in ("AGENTS.md", "README.md", "CHANGELOG.md", "pyproject.toml", ".gitignore", "MANIFEST.in"))
    destination.mkdir(parents=True)
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(source)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/"EXPORT_MANIFEST.json").write_text(json.dumps(dict(schema_version=1, sha256=hashes), indent=2)+"\n")
    print(f"Exported {len(hashes)} source/documentation files to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    curate(parser.parse_args().destination)
