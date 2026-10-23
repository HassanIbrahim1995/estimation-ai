# Specialist strategy review — 24 October 2026

## What the rules establish

Pocket Estimation's official tutorial states that the goal is to achieve the
estimated number of tricks, the winner leads the next trick, and players must
follow suit if possible. It defines over/under rounds and describes taking
tricks sooner in over rounds and avoiding excess tricks in under rounds.
These are useful tendencies, not unconditional action rules.

Primary source inspected for this review:
https://blog.pocketestimation.com/en-how-to-play-estimation/

The lessons below are derived engineering examples, not quotations or claims
that Pocket's authors endorse this implementation. Scoring ambiguities in
RULES.md remain unresolved.

## Exact lesson: the same cards require opposite decisions

Two tricks remain. You play fourth, holding QC and AC. KC, 5C and 6C
are on the table; each opponent has one of 2C, 3C, 4C left. No trump.

| Remaining target | Play | Why |
|---|---|---|
| One trick | QC | Lose now; AC must win the final trick. |
| Two tricks | AC | Win now, then QC wins against all remaining lower clubs. |

The outcome does not depend on which opponent holds which small club.
The tests supply a complete card history consistent with eleven previous
tricks and check both recommendations across several sampled allocations.

Added tests in `tests/test_strategy.py`:

- Duck now when exactly one trick is needed.
- Take now when both remaining tricks are needed.
- Verify the exact terminal trick counts under every sampled allocation used.

All three passed. These test the teacher's strategic conclusions, not merely
whether a particular feature number is present.

## Teaching priorities

1. **Count unavoidable future winners.** A safe win now can create an unwanted
   forced winner later. Include paired positions with identical cards and
   different targets, as above.
2. **Treat zero as a continuing escape problem.** Prefer examples where a high
   card can be discarded on a higher lead; do not teach “always play low.”
3. **Keep exit cards.** When the target is reached, losing the current trick
   only helps if the subsequent hand can also avoid winning. Teach two- and
   three-trick continuations before adding noisy long-horizon examples.
4. **Value trump in context.** Ruffing earns a trick but consumes control;
   declining to ruff can preserve a later entry. Label through continuations.
5. **Use public voids as hard constraints.** Failing to follow establishes a
   missing suit. An auction bid is weaker evidence and should influence a
   calibrated distribution, not eliminate otherwise possible hands.
6. **Separate target completion from relative-score tactics.** Denying another
   player may improve score advantage but may also destroy one's own contract.
   Record both contract outcome and score advantage for lesson explanations.
7. **Keep match position separate.** A risk appropriate when trailing late in
   a match may be poor while leading. The present features ignore match scores;
   this is an explicit future extension, not a capability already learned.

## Concrete limitations found in code

- `Teacher.values` computes paired standard errors, but `Teacher.act` directly
  selects the largest estimated mean. Therefore the live teacher is not itself
  confidence-gated. A downstream training filter does not change that fact.
- `reconstruct` creates a default `Game`; observations do not carry `Rules`.
  Non-default scoring is consequently lost in rollouts. Restrict these results
  to the default profile until rule identity is represented and checked.
- The hand sampler respects counts and voids but ignores auction likelihoods
  and is not a uniform posterior. More samples do not remove this bias.
- A 22-feature policy cannot distinguish all strategically different positions.
  Adding a neural layer cannot recover information absent from those features.
- Rollout policies are deterministic and share one policy. Teacher confidence
  estimates describe sampled-hand uncertainty, not uncertainty about opponent
  styles or rollout-policy mistakes.

## Promotion requirements

Keep the established model as a frozen reference. Use separate deals for
lesson fitting, checkpoint selection, gameplay validation and final reporting.
Require improvement on gameplay validation as well as the lesson metric.
Report exact-contract rates by curriculum category, match win credit against
multiple opponents, seed-cluster uncertainty, and decision time. Deterministic
lessons are regression checks; passing them alone does not establish stronger
human play.
