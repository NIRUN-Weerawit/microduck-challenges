# Microduck challenges

One folder per challenge open on the Microduck Arena. Each is a complete,
working training environment: train it unchanged and the robot competes.

| Folder | Arena event | Task |
|---|---|---|
| sprint_2m | sprint-2m | Mjlab-Sprint2m-MicroDuck |
| got_a_talent | got-a-talent | Mjlab-GotATalent-MicroDuck |
| hill_climb_1m5 | hill-climb-1m5 | Mjlab-HillClimb1m5-MicroDuck |
| roller_sprint_2m | roller-sprint-2m | Mjlab-RollerSprint2m-MicroDuck |

Training with a coding agent (Claude Code, Codex, Cursor…)? Each challenge folder has an
`AGENTS.md` that tells it which files to change, how to run and judge a training, and the
challenge's rules; point the agent at it.

MicroDuck Got a Talent is a performance, not a race: the Arena plays the entry's
`timeline.json` for up to 20 s on an open stage, and visitors vote.
`got_a_talent/timeline.json` is Pollen's: `twist[0]` is the sit flag (1 sits,
0 stands) and `head` is 4 deltas in radians from the default pose (neck_pitch,
head_pitch, head_yaw, head_roll). Check a policy against a timeline with
`uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.check policy.onnx [--timeline FILE]`,
and publish the timeline with the policy: `uv run publish … --timeline src/microduck_challenges/got_a_talent/timeline.json`.

## The workshop

Each challenge's page on the Arena has a **Train my own robot** button with one line to copy:

```bash
uv run https://pollen-robotics-microduck-arena.hf.space/workshop.py sprint-2m
```

It works in the clone you run it in (or clones this repo next to you) and opens the workshop in your
browser, on that challenge. Its home says what is set up (Hugging Face Jobs billing, your fork, an
NVIDIA GPU) and lists the challenges. A challenge's page says which files to change, commits them
before each training, trains on your NVIDIA GPU or on Hugging Face Jobs, and keeps one ticket per
training: every checkpoint is evaluated with the Arena's own rules as it arrives. A training's page
replays any checkpoint, and **Publish and compete** publishes the policy to your Hugging Face account
**and enters** it on the Arena, then opens its result. On MicroDuck Got a Talent, a training's page
also edits `timeline.json`, your performance, and plays it with the checkpoint you pick. From a clone,
`uv run --group workshop workshop` opens the home and `uv run --group workshop workshop sprint-2m`
opens one challenge.

To allow multiple local runs from the workshop, apply the included compatibility
patch after installing dependencies, then restart the workshop:

```bash
uv run --group workshop python scripts/enable_parallel_training.py
uv run --group workshop python scripts/enable_warm_start.py
uv run --group workshop workshop sprint-2m
```

Runs retain separate tickets, reward settings and launch commits. Launch requests
are serialized, but training processes can overlap and share the GPU. Use distinct
run names and size environment counts to fit available GPU memory. Reapply the
patches if an Arena package update replaces the installed files. On the New
training form, **Start from** lists this challenge's local checkpoints. Select
one to load its policy and start a new run at iteration 0, with curricula
restarted. This uses the trainer's `MICRODUCK_WARM_START=1` support; the
checkpoint's optimizer and normalizer are loaded as well. Warm start is
available for local training only. The source checkpoint is kept untouched.

## Train from a shell

Needs Python 3.12 and `uv`. Training needs an NVIDIA GPU, or Hugging Face Jobs.

```bash
git clone https://github.com/pollen-robotics/microduck-challenges && cd microduck-challenges
uv sync --group workshop
uv run hf auth login   # once: Hugging Face Jobs and publish run under your account
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.seed 1 --agent.logger tensorboard --agent.run-name first
# no GPU: add --hf-jobs --no-wandb
```

A challenge's knobs are in its `challenge.toml` (`[params]`); its whole
environment is in `env.py`. Edit either, commit, train. To add a custom
reward, write the function in a `rewards.py` next to the challenge's `env.py`
and add it to `cfg.rewards` in `env.py`.

`--hf-jobs` trains on Hugging Face's infrastructure instead of a local GPU;
nothing lands in `logs/rsl_rl/` here. Its checkpoints go to the Hub repo the
submission prints (`[ckpt] checkpoints -> <repo>`), mirroring `logs/rsl_rl/` —
bring them back before publishing:

```bash
uv run hf download <that repo> --local-dir logs/rsl_rl
```

Then publish and enter it in one command:

```bash
uv run publish --run logs/rsl_rl/<folder>/<run> --repo <you>/microduck-<name> --no-private --enter
```

It uploads the policy to your Hugging Face account, enters it on the
challenge's event on the Arena, and prints its time and the page of its result.
`publish` creates a new Hub repo private by default, and the Arena reads public
repos only: hence `--no-private`.

## Tests

```bash
uv run --group workshop pytest
```

(the tests import the Arena, which lives in the `workshop` group).
