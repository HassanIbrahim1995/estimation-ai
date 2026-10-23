# Estimation AI: specialist teaching round — 24 October 2026

## Outcome

The new neural candidate was **not promoted**. The default `models/trained.json`
remains the previous self-play player. The new candidate improved agreement
with the sampled teacher, but separate gameplay validation did not establish
a reliable score improvement. This is a research prototype, not a proven
human-expert player or certified Pocket Estimation replica.

## Work completed

Three specialist agents reviewed rules, strategy and training methodology.
Their findings are in RULES_REVIEW.md and STRATEGY_REVIEW.md; the training
review produced uncertainty/provenance checks, duplicate-deal rejection,
a zero-update checkpoint option, and a paired gameplay promotion gate.
The rule profile remained fixed throughout this comparison.

Generated 1,024 fresh independent-deal positions: 256 each of under rounds,
over rounds, Dash Calls and exactly-target-reached positions. Each legal
move was evaluated in 16 plausible hidden deals, respecting public voids,
card counts and the player's own hand. Actual opponent cards are excluded.
These labels reflect one rollout policy and a biased feasible-hand sampler;
they are estimates, not optimal-play ground truth.

Training used 819 examples; 205 separate examples selected a checkpoint over
200 epochs. The selected epoch was 20. The network has 768 parameters.
127 examples supported a correction under the uncertainty screen.

## Results

| Measure | Previous model | New candidate |
|---|---:|---:|
| Held-out teacher regret (lower is better) | 5.175 | 4.918 |
| Fresh full-match win credit vs three heuristics | 35.0% | 33.5% |
| Exact-bid rate in those matches | 43.1% | 41.0% |

Gameplay validation used 144 seed blocks, four seats each: 576 deals per model.
Candidate score-advantage improvement: 0.158; approximate
95% paired confidence interval [-1.3047521328311609, 1.6207243550533827].
The interval includes zero, so the promotion gate rejected the candidate.

The final report used fresh seeds 920000–920059, 240 full 18-round matches
per model (480 total), against three heuristic players. These seeds did not
select checkpoints. Paired match-score advantage change:
-9.997, approximate 95% interval
[-24.480363022211293, 4.4859185777668475]. This comparison is descriptive;
it does not override the prior promotion decision or justify tuning on these seeds.
Full metrics and seed-block results: reports/comparison_v2.json.

## Concrete lesson and remaining weakness

With two tricks left, fourth to play in no-trump, hold QC and AC against
KC/5C/6C on the table; opponents each have one lower club left.
Need one trick: play QC and win the final trick with AC.
Need two tricks: play AC, then win the final trick with QC.
The teacher passes both cases across hidden allocations. Both learned models
still incorrectly play AC in the one-trick case. Thus teacher correctness
does not establish that its student learned every lesson.

All 25 tests passed, covering legality, scoring consistency, hidden-card
isolation, teacher reconstruction, exact endgames and uncertainty handling.
Tests ran on Linux; the Windows launcher was not executed here.

## Reproduce

```bash
python -m estimation.curriculum --examples 1024 --samples 16 --seed 4200000 --output reports/curriculum_v2.jsonl
python -m estimation.teach_neural --data reports/curriculum_v2.jsonl --epochs 200 --validation-seed-start 740000 --validation-seeds 144 --output models/neural_v2.json --report reports/neural_v2.json
python -m estimation.compare --seed-start 920000 --seeds 60
python -m unittest discover -s tests -v
python -m estimation play
```

Do not reuse final seeds to tune a later model. The generation seed and dataset
fingerprint are recorded in the command and reports/neural_v2.json.

## Next training objective

Add paired short-endgame curricula and richer future-winner/exit-card features,
then test on new suit/rank permutations and fresh game seeds. Preserve the
existing champion until validation supports a replacement. Non-default rules
are not carried through teacher reconstruction; this run uses default rules only.

## Sources

Primary rule source: https://blog.pocketestimation.com/en-how-to-play-estimation/

Arabic tutorial, including ambiguities documented in the rules review:
https://blog.pocketestimation.com/how-to-play-estimation/

Comparison variant, not silently merged:
https://blog.jawaker.com/en/estimation-rules-en/

All numerical results above come from this project's saved run reports.
