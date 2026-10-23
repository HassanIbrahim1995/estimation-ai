# Specialist rules audit — 24 October 2026

Scope: read-only audit of `estimation/game.py` and `RULES.md`. No engine rules
were changed. This review does **not** certify exact Pocket compatibility.

## Findings

The implementation is internally consistent with its declared provisional
profile. No newly confirmed Pocket-rule violation was established by the public
tutorial. Several choices still change the training reward or legal actions and
must remain explicit uncertainties.

### Confirmed foundation

Pocket's English tutorial supports four independent players, 13 cards each,
counterclockwise play, following suit, minimum four-trick auction, NT/S/H/D/C
bid priority, caller leading, estimates capped at the caller's bid, equal bids
marked With, and a total other than 13. It also describes 18 rounds, the final
five fixed suits with an 8+ override, and all-pass advancing to a doubled round.
The engine implements these mechanics [1].

### Highest-priority uncertainties

| Issue | Current implementation | Audit result / next evidence |
|---|---|---|
| Ordinary failed bid | −10 minus absolute bid error | Public score table's “Tricks Amount” does not distinguish bid, tricks taken, or error. Capture an app result with bid 5 and actual 2. |
| Dash scoring | Replaces the ordinary base; sole-result and risk can stack | Neither text nor row-based table settles replacement versus addition. Capture both successful and failed Dash, with more than one winner/loser. |
| All players fail | Zero score, then doubles | Explicitly borrowed from Jawaker [3], not established in the Pocket tutorial. |
| Risk on over rounds | Applied symmetrically above/below 13 | English tutorial supports both directions; Arabic description mentions under totals and 11 specifically [1,2]. Keep this source disagreement visible. |
| Risk cap | At most six bonuses | `max_risk=6` is an implementation choice without a cited Pocket cap. `RULES.md` exposes the field but does not flag the cap itself in its decision table. |
| Zero-call limit | At most two Dash Calls; ordinary zero estimates unlimited | Jawaker limits Dash/Dash Call together to two [3]. Pocket text does not settle its own limit. This is a deliberate variant difference, not proof of a Pocket bug. |
| Auction pass and starting seat | Passing exits; dealer's next seat starts | Public Pocket tutorial lacks enough detail to certify repeat-turn semantics and initial seat. |
| Fixed-round ties and super-call timing | First highest estimate wins; separate super auction first | Public Pocket text confirms highest estimate and override threshold, not these tie/timing details. |
| Pre-auction void declaration | Omitted | Jawaker describes Avoid [3]; do not import it into Pocket without evidence. |

The English/Arabic Risk discrepancy is newly highlighted by this audit. Both
pages are publisher-owned tutorials, so choosing one silently would conceal a
real ambiguity. The English interpretation is already the selected profile.

## Code-derived edge cases worth preserving as regression scenarios

These are deductions from implementation, not claims about the real app:

* Fixed rounds allow all four estimates to be zero. This selects the first
  estimating-order maximum as caller and classifies non-Dash zero bidders as
  With. With two prior Dash Calls, a Dash seat can also win that zero tie. Check
  the app before teaching this rare pattern as normal play.
* Extremely high over totals can exceed a six-Risk difference calculation, but
  scoring clips them. For bids `[13, 13, 13, 12]`, the uncapped integer formula
  gives 19 Risk units, while the implementation gives six. This example is a
  boundary check, not a recommendation for ordinary bidding.
* A skipped all-pass deal still consumes a round index. A skip at round 13
  therefore advances to the first fixed-suit round. The tutorial's wording is
  compatible with this, but an app round-counter example would establish it.
* Round-score optimization can prefer causing opponents to fail over protecting
  one's own exact bid. That is an intentional reward choice, not a rules defect;
  its relationship to full-match win probability needs separate evaluation.

## Recommended training policy

Keep the current profile identifier and checkpoint provenance. Do not change
scoring while comparing old/new policies. Record rule parameters in training
reports; label all benchmarks as applying to this provisional profile. Resolve
ordinary miss and Dash stacking first because these affect almost every
training run. Existing unit tests can establish implementation consistency,
but cannot establish proprietary app parity.

## Sources

1. Pocket Estimation, English publisher tutorial, checked 24 October 2026:
   https://blog.pocketestimation.com/en-how-to-play-estimation/
2. Pocket Estimation, Arabic publisher tutorial, checked 24 October 2026:
   https://blog.pocketestimation.com/how-to-play-estimation/
3. Jawaker, Estimation Rules, comparison variant, checked 24 October 2026:
   https://blog.jawaker.com/en/estimation-rules-en/

The score image already present at `../score.jpg` was visually inspected as
supporting project material. Its provenance was not independently re-established
in this audit: the live English tutorial's linked image fetch failed. No new
claim of exact image provenance is made.
