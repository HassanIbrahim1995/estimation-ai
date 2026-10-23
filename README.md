# Estimation AI — trained local prototype

A working Python simulator, a trainable player, a saved learned model, held-out
benchmarks and a terminal game for one human against three bots. Runs locally;
no API key, LLM, cloud inference or human-game dataset is needed.

**Latest work:** see RESULTS.md for the specialist review, fresh teaching run,
and promotion decision. The optional candidate uses a small residual neural layer;
the default player remains the validated self-play checkpoint.

**Status:** research prototype, not a verified expert or an exact clone of
Pocket Estimation. Read RULES.md: some app rules remain unverified. The learned
model is a compact linear action scorer trained by evolutionary league
self-play, **not** a deep neural network. Do not interpret simulation volume
as a measure of human playing strength.

## Run on Windows

Install Python 3.11+ if needed: https://www.python.org/downloads/windows/
Unzip, open PowerShell inside the `estimation-ai` folder, then:

Alternatively, double-click `play-windows.bat` after installing Python. This
creates a local virtual environment and installs NumPy on first launch. The
batch launcher is provided for convenience; execution was tested on Linux,
not on a Windows machine.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m estimation play
```

No PowerShell activation or execution-policy change is needed. Enter the
displayed **option number**, not the card ID. You play seat 0.

## Run on Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m estimation play
```

In the commands below use your virtual environment's Python executable.

```bash
python -m estimation demo --seed 42
python -m unittest discover -s tests -v
python -m estimation.evaluate --seeds 40
python -m estimation.train --generations 100 --population 24 --deals 144 --output models/next.json
python -m estimation.train --resume models/trained.json --output models/next.json
```

`--resume` starts a new search from the checkpoint's weights; it does not restore
the optimizer RNG, league or variance. Training saves the selected model after
each generation. Keep separate output names to preserve the shipped model.
Training history is written to `reports/training.json` and will be overwritten.
The longer command is an experiment, not a promise of stronger play.

## What the bot knows

Its cards; public bids and auction history; played cards; inferred void suits;
each player's required and achieved tricks; trump, round and scores. The
observation interface excludes opponents' hands. Action features use card
memory, approximate chances of winning a trick, suit length, remaining targets,
and whether an opponent needs one more trick or has already reached their bid.
The current model does not use all exposed context: match scores and auction
history are available but not encoded in its learned action features yet.

The engine enforces legal moves independently. The bot cannot revoke, inspect
the shuffle seed or request a different hand. Its probability calculation is
an approximation, not a calibrated belief model. Expert human bluffing and
opponent adaptation are not established.

## Training and evaluation

The initial player is a hand-written heuristic. Cross-entropy optimization
mutates its 31 scoring / bidding parameters and selects candidates using
terminal score advantage against frozen baseline and historical policies.
No supervised labels or hidden-card features enter the model. It learns both
bidding calibration and card selection. Default Dash Call behavior is cautious;
rare events need focused training beyond this initial run.

Training deals use seeds below 600000; checkpoint selection uses 700000–700143.
Final evaluation starts at 900000 and rotates the candidate through all four
seats on the same match seeds. Each match has 18 rounds. Reported confidence
intervals cluster the four seat rotations by seed. Ties split win credit.
The random baseline uses random **card play**, with heuristic bidding.
Results apply only to these bots and this provisional rule profile.

The initial diagnostic used 900000–900059. After diagnosing overly cautious
baseline bidding, the baseline was recalibrated and the model retrained;
the shipped final benchmark uses **910000–910059**. The initial run is retained
under `reports/initial_diagnostic` for transparency, not as a final claim.

To reproduce the shipped training run from the revised baseline:

```bash
python -m estimation.train --generations 30 --population 12 --deals 72 --seed 20261024
python -m estimation.evaluate --seeds 60 --seed-start 910000
```

See `reports/training.json`, `reports/evaluation.json` and `RESULTS.md` for the
actual run. The evaluation set must remain untouched during future model
selection; choose new final seeds for subsequent experiments.

## Integrate with a Java / Spring Boot game

`Game.observe().to_dict()` is the public observation schema. Card IDs are
`suit * 13 + rank`, suits C/D/H/S = 0/1/2/3, ranks 2..A = 0..12; trump 4 means NT.
Seat IDs increase counterclockwise. Auction action 0 means pass; otherwise
`1 + (tricks - 4) * 5 + trump`. Estimate actions are 0..13. Dash actions are 0/1.
Card-play actions are card IDs.

```bash
python -m estimation predict --model models/trained.json
```

This reads one public observation JSON per stdin line and writes
`{"action": ...}` per output line. A Java process can keep it running via
`ProcessBuilder`; the game server must validate the returned action. No server
or authentication layer is included. The observation's `legal` values must be
provided by a trusted rules engine, not a user's client.

## Work needed for an expert player

1. Resolve the rule questions in RULES.md against in-app examples.
2. Add a history encoder and calibrated hidden-hand sampler respecting public
   voids, card counts and bidding information.
3. Train a recurrent neural policy/value model across all phases and full
   matches, against a diverse league; retain this engine as the legality oracle.
4. Evaluate belief-sampled lookahead search for close decisions. Avoid
   perfect-information rollout policies that accidentally exploit sampled cards.
5. Use a separate human test league, unseen seeds, different opponent styles,
   latency limits and confidence intervals before calling it expert.

Self-play is supported by research in other imperfect-information card games;
it does not automatically transfer their performance or theoretical guarantees
to four-player Estimation [3–5]. A GPU-backed neural training run is not included
in this release.

## Sources

1. Pocket Estimation official tutorial:
   https://blog.pocketestimation.com/en-how-to-play-estimation/
2. Jawaker official rules (a comparison variant, not identical):
   https://blog.jawaker.com/en/estimation-rules-en/
3. Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in
   Imperfect-Information Games* (2016): https://arxiv.org/abs/1603.01121
4. Zha et al., *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
   Learning*, ICML 2021: https://proceedings.mlr.press/v139/zha21a.html
5. Brown et al., *Combining Deep Reinforcement Learning and Search for
   Imperfect-Information Games* (2020): https://arxiv.org/abs/2007.13544

Independent project; no Pocket Estimation code, assets or accounts are used.
