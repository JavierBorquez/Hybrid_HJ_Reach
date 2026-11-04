"""Portable, non-pickle results. Numerical data and JSON metadata are separate."""
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np


@dataclass
class Result:
    kind: str
    coordinates: tuple
    horizons: object
    values: object
    mode_names: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.coordinates = tuple(np.asarray(v) for v in self.coordinates)
        self.horizons = np.asarray(self.horizons)
        if self.kind not in ("reach", "avoid", "reach_avoid"):
            raise ValueError("Unknown problem kind")
        if (self.horizons.ndim != 1 or len(self.horizons) == 0
                or not np.isfinite(self.horizons).all() or self.horizons[0] < 0
                or np.any(np.diff(self.horizons) <= 0)):
            raise ValueError("Expected increasing nonnegative saved horizons")
        for axis in self.coordinates:
            if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError("Expected finite increasing coordinates")
        expected = (len(self.horizons), len(self.mode_names)) + tuple(len(v) for v in self.coordinates)
        if self.values.shape != expected:
            raise ValueError(f"Expected values shape {expected}")

    def save(self, directory):
        """Save history without compression or pickle; reload supports mmap.

        Caller chooses a new directory to prevent silently mixing experiment runs.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "values.npy", np.asarray(self.values), allow_pickle=False)
        self._write_metadata(directory)

    def _write_metadata(self, directory):
        directory = Path(directory)
        np.savez(directory / "axes.npz", horizons=self.horizons,
                 **{f"axis_{i}": v for i, v in enumerate(self.coordinates)})
        record = dict(schema_version=1, kind=self.kind, mode_names=list(self.mode_names),
                      ndim=len(self.coordinates), metadata=self.metadata)
        (directory / "metadata.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory, *, mmap_mode="r"):
        directory = Path(directory)
        record = json.loads((directory / "metadata.json").read_text())
        if record["schema_version"] != 1:
            raise ValueError("Unsupported result schema")
        with np.load(directory / "axes.npz", allow_pickle=False) as axes:
            coordinates = tuple(axes[f"axis_{i}"] for i in range(record["ndim"]))
            horizons = axes["horizons"]
        return cls(record["kind"], coordinates, horizons,
                   np.load(directory / "values.npy", mmap_mode=mmap_mode, allow_pickle=False),
                   tuple(record["mode_names"]), record["metadata"])
