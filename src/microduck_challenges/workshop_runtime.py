"""Freeze workshop launch metadata before task registration in each child."""

import json
import os
from pathlib import Path

SNAPSHOT_ENV = "MICRODUCK_WORKSHOP_LAUNCH_SNAPSHOT"


def apply_launch_snapshot():
    raw = os.environ.get(SNAPSHOT_ENV)
    if not raw:
        return
    snapshot = json.loads(raw)
    target = Path(snapshot["toml"]).resolve()
    values = {name: float(value) for name, value in snapshot["params"].items()}
    from mjlab_microduck import challenge, provenance

    original_params = challenge.params
    original_checkout = provenance.checkout

    def frozen_params(tasks_file):
        path = Path(tasks_file).resolve().parent / challenge.FILE
        return dict(values) if path == target else original_params(tasks_file)

    def frozen_checkout(cwd=None):
        root = Path(cwd or Path.cwd()).resolve()
        if root == Path(snapshot["root"]).resolve():
            return dict(snapshot["provenance"])
        return original_checkout(cwd)

    challenge.params = frozen_params
    provenance.checkout = frozen_checkout
