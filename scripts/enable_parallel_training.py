"""Enable concurrent local launches in the installed Microduck Arena workshop.

Run after installing/updating dependencies, then restart the workshop. Validate
both source transformations before writing either file; retain original backups.
"""

import importlib.util
from pathlib import Path

GUARD = '''            if any(t.where == "local" and t.state == "running" for t in ws.trainings.values()):
                raise HTTPException(status_code=409, detail="A training already runs on this machine: stop it first.")
'''
ALLOW = "            # Parallel local runs enabled; the launch lock still serializes settings/commits.\n"
SPAWN = '''        self.proc = subprocess.Popen(
'''
SNAPSHOT = '''        # Freeze settings and provenance before another launch edits the checkout.
        launch_env = child_env()
        if where == "local":
            from mjlab_microduck.provenance import checkout
            launch_env["MICRODUCK_WORKSHOP_LAUNCH_SNAPSHOT"] = json.dumps({
                "root": str(root.resolve()), "toml": str(challenge.toml.resolve()),
                "params": {k: v["value"] for k, v in
                           tomllib.loads(challenge.toml.read_text())["params"].items()},
                "provenance": checkout(root),
            })
        self.proc = subprocess.Popen(
'''
CHILD_ENV = 'text=True, errors="replace", bufsize=1, env=child_env(), start_new_session=True,'
FROZEN_ENV = CHILD_ENV.replace("env=child_env()", "env=launch_env")


def transform_sources(app, training):
    if ALLOW not in app:
        if app.count(GUARD) != 1:
            raise ValueError("Unsupported workshop launch guard; no files changed.")
        app = app.replace(GUARD, ALLOW)
    if 'MICRODUCK_WORKSHOP_LAUNCH_SNAPSHOT' not in training:
        if training.count(SPAWN) != 1 or training.count(CHILD_ENV) != 1:
            raise ValueError("Unsupported workshop Training implementation; no files changed.")
        training = training.replace("import os\n", "import os\nimport json\nimport tomllib\n", 1)
        training = training.replace(SPAWN, SNAPSHOT).replace(CHILD_ENV, FROZEN_ENV)
    compile(app, "workshop/app.py", "exec")
    compile(training, "workshop/training.py", "exec")
    return app, training


def main():
    spec = importlib.util.find_spec("arena")
    if spec is None or spec.origin is None:
        raise SystemExit("Install the workshop group first: uv sync --group workshop")
    root = Path(spec.origin).parent / "workshop"
    paths = [root / "app.py", root / "training.py"]
    originals = [path.read_text() for path in paths]
    changed = transform_sources(*originals)
    for path, before, after in zip(paths, originals, changed, strict=True):
        if before == after:
            continue
        backup = path.with_suffix(path.suffix + ".parallel-training.bak")
        if not backup.exists():
            backup.write_text(before)
        path.write_text(after)
    print("Parallel local training enabled. Restart the workshop to activate it.")


if __name__ == "__main__":
    main()
