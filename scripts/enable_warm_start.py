"""Add local checkpoint warm-start controls to the installed Arena workshop.

Run after ``uv sync --group workshop`` (and after enable_parallel_training.py),
then restart the workshop. The installed Arena wheel is not tracked by Git, so
this installer is kept in the checkout and can be rerun after an update.
"""

import importlib.util
from pathlib import Path


def replace_once(source: str, before: str, after: str, label: str) -> str:
    if after in source:
        return source
    if source.count(before) != 1:
        raise ValueError(f"Unsupported {label}; no files changed.")
    return source.replace(before, after, 1)


def transform_app(source: str) -> str:
    source = replace_once(
        source,
        '        "gpu": has_gpu(),\n',
        '        "gpu": has_gpu(),\n        "warm_start_supported": True,\n',
        "workshop feature status",
    )
    source = replace_once(
        source,
        'class Launch(BaseModel):\n',
        'class WarmStart(BaseModel):\n'
        '    run_id: str\n'
        '    checkpoint: int\n\n\n'
        'class Launch(BaseModel):\n',
        "workshop launch model",
    )
    source = replace_once(
        source,
        '    params: dict[str, float] = {}\n\n\n@router.post("/challenges/{event}/train")',
        '    params: dict[str, float] = {}\n'
        '    warm_start: WarmStart | None = None\n\n\n@router.post("/challenges/{event}/train")',
        "workshop launch fields",
    )
    source = replace_once(
        source,
        '        if git(ws.root, "status", "--porcelain"):\n',
        '        warm_start = None\n'
        '        if body.warm_start is not None:\n'
        '            if body.where != "local":\n'
        '                raise HTTPException(status_code=400, detail="Warm start is available for local training only.")\n'
        '            source_run = next((r for r in find_runs(ws.root, challenge.task)\n'
        '                               if r.id == body.warm_start.run_id), None)\n'
        '            if source_run is None or body.warm_start.checkpoint not in source_run.checkpoints:\n'
        '                raise HTTPException(status_code=400, detail="Choose an existing checkpoint from this challenge.")\n'
        '            warm_start = (source_run.id, body.warm_start.checkpoint)\n'
        '        if git(ws.root, "status", "--porcelain"):\n',
        "workshop warm-start validation",
    )
    source = replace_once(
        source,
        '            ws.root, challenge, body.name, body.where, body.recipe, body.params, hf_user)\n',
        '            ws.root, challenge, body.name, body.where, body.recipe, body.params, hf_user, warm_start)\n',
        "workshop training call",
    )
    compile(source, "arena/workshop/app.py", "exec")
    return source


def transform_training(source: str) -> str:
    source = replace_once(
        source,
        'def train_command(challenge, name: str, recipe: dict[str, int], hf_user: str | None) -> list[str]:',
        'def train_command(challenge, name: str, recipe: dict[str, int], hf_user: str | None,\n'
        '                  warm_start: tuple[str, int] | None = None) -> list[str]:',
        "workshop command signature",
    )
    source = replace_once(
        source,
        '    if hf_user:\n        command += ["--hf-jobs",',
        '    if warm_start is not None:\n'
        '        run_id, checkpoint = warm_start\n'
        '        command += ["--agent.resume", "True",\n'
        '                    "--agent.load-run", f"^{re.escape(run_id)}$",\n'
        '                    "--agent.load-checkpoint", rf"^model_{checkpoint}\\.pt$"]\n'
        '    if hf_user:\n        command += ["--hf-jobs",',
        "workshop command construction",
    )
    source = replace_once(
        source,
        '                 params: dict[str, float], hf_user: str | None = None):',
        '                 params: dict[str, float], hf_user: str | None = None,\n'
        '                 warm_start: tuple[str, int] | None = None):',
        "workshop training signature",
    )
    source = replace_once(
        source,
        '        command = train_command(challenge, name, recipe, self.namespace)\n',
        '        command = train_command(challenge, name, recipe, self.namespace, warm_start)\n',
        "workshop command call",
    )
    source = replace_once(
        source,
        '        launch_env = child_env()\n',
        '        launch_env = child_env()\n'
        '        if warm_start is not None:\n'
        '            launch_env["MICRODUCK_WARM_START"] = "1"\n',
        "workshop child environment",
    )
    source = replace_once(
        source,
        '                     started=datetime.now(UTC).isoformat(timespec="seconds"), **hf)\n',
        '                     started=datetime.now(UTC).isoformat(timespec="seconds"),\n'
        '                     warm_start={"run_id": warm_start[0], "checkpoint": warm_start[1]}\n'
        '                     if warm_start is not None else None, **hf)\n',
        "workshop launch record",
    )
    compile(source, "arena/workshop/training.py", "exec")
    return source


def transform_frontend(source: str) -> str:
    source = replace_once(
        source,
        '      <label class="field">Run name <input name="run" spellcheck="false" maxlength="40" /></label>\n',
        '      <label class="field">Run name <input name="run" spellcheck="false" maxlength="40" /></label>\n'
        '      <label class="field">Start from <select name="warm-start"><option value="">Fresh policy</option></select></label>\n'
        '      <p class="hint warm-hint">Warm start loads the selected policy, then restarts the training schedule.</p>\n',
        "workshop training form",
    )
    source = replace_once(
        source,
        'function Mt({challenge:e,gpu:t},{onCommit:n,onTrain:o})',
        'function Mt({challenge:e,gpu:t,warmStart:w},{onCommit:n,onTrain:o})',
        "workshop form capabilities",
    )
    source = replace_once(
        source,
        's(".gate__identity pre").textContent=Nt,',
        's("select[name=warm-start]").closest("label").hidden=!w,'
        's(".warm-hint").hidden=!w,s(".gate__identity pre").textContent=Nt,',
        "workshop warm-start visibility",
    )
    source = replace_once(
        source,
        'function C(){const{code:p,account:d}=m,',
        'function Q(p){const d=s("select[name=warm-start]"),f=d.value,'
        'b=[new Option("Fresh policy","")];for(const v of p)if(v.run_id)'
        'for(const j of v.checkpoints??[])b.push(new Option(`${v.name} · checkpoint ${j.n}`,'
        '`${v.run_id}:${j.n}`));d.replaceChildren(...b);if(b.some(v=>v.value===f))d.value=f}'
        'function C(){const{code:p,account:d}=m,',
        "workshop checkpoint selector",
    )
    source = replace_once(
        source,
        'return{name:p("run").trim(),where:l,recipe:',
        'const d=p("warm-start"),f=d.split(":"),b=w&&l==="local"&&d?'
        '{run_id:f[0],checkpoint:Number.parseInt(f[1],10)}:null;'
        'return{name:p("run").trim(),where:l,warm_start:b,recipe:',
        "workshop warm-start request",
    )
    source = replace_once(
        source,
        's(".train-go").disabled=f||y||!!b}',
        's("select[name=warm-start]").disabled=!w||l!=="local";s(".train-go").disabled=f||y||!!b}',
        "workshop local selector state",
    )
    source = replace_once(
        source,
        'update(p){m=p,C()},reseed:g}',
        'update(p){m=p,Q(p.jobs??[]),C()},reseed:g}',
        "workshop form refresh",
    )
    source = replace_once(
        source,
        'function b(){f.update({code:m,account:n.get()})}b();',
        'let q=[];function b(){f.update({code:m,account:n.get(),jobs:q})}b();',
        "workshop jobs state",
    )
    source = replace_once(
        source,
        'const f=Mt({challenge:h,gpu:t.gpu},{onCommit:',
        'const f=Mt({challenge:h,gpu:t.gpu,warmStart:t.warm_start_supported===!0},{onCommit:',
        "workshop feature input",
    )
    source = replace_once(
        source,
        'm=I,b();const x=j.filter(',
        'm=I,q=j,b();const x=j.filter(',
        "workshop jobs refresh",
    )
    return source


def transform_html(source: str) -> str:
    return replace_once(
        source,
        '/assets/workshop-CIygtbAu.js"',
        '/assets/workshop-CIygtbAu.js?warm-start=1"',
        "workshop script URL",
    )


def main() -> None:
    spec = importlib.util.find_spec("arena")
    if spec is None or spec.origin is None:
        raise SystemExit("Install the workshop group first: uv sync --group workshop")
    root = Path(spec.origin).parent
    paths = [root / "workshop" / "app.py", root / "workshop" / "training.py",
             root / "frontend" / "assets" / "workshop-CIygtbAu.js",
             root / "frontend" / "workshop.html"]
    transforms = [transform_app, transform_training, transform_frontend, transform_html]
    originals = [path.read_text() for path in paths]
    changed = [transform(before) for before, transform in zip(originals, transforms, strict=True)]
    for path, before, after in zip(paths, originals, changed, strict=True):
        if before == after:
            continue
        backup = path.with_suffix(path.suffix + ".warm-start.bak")
        if not backup.exists():
            backup.write_text(before)
        path.write_text(after)
    print("Checkpoint warm start enabled. Restart the workshop and refresh its page.")


if __name__ == "__main__":
    main()
