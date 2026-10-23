# Rule profile: pocket-inspired-provisional-v1

## Confirmed core from Pocket's tutorial

Four independent players receive 13 cards each. Play is counterclockwise;
follow suit when possible. Bidding starts at four tricks; equal-sized bids
rank NT > spades > hearts > diamonds > clubs. The caller leads. Others estimate
up to the caller's bid; equal bids are With. Estimates cannot total 13.
Dash Call commits to zero before auction. A match has 18 rounds; the last five
use NT/S/H/D/C unless overridden by an 8+ bid. All-pass doubles the next round.

Source: https://blog.pocketestimation.com/en-how-to-play-estimation/

## Explicit provisional decisions

| Detail | Implemented choice | Status |
|---|---|---|
| Failed round | −10 minus absolute bid error | Pocket table has −10; meaning of “Tricks Amount” needs app verification |
| Successful ordinary round | 10 + bid | Tutorial table |
| Caller / With | ±10 | Tutorial table |
| Sole winner / loser | ±10 | Tutorial table |
| Risk | ±10 per two tricks away from 13, awarded to last estimator | Stacking and eligibility edge cases need checking |
| Dash Call | ±25 over, ±33 under, replacing base; sole-result/risk still stack | Stacking needs checking |
| No exact bids | Zero scores; next round doubles | Adopted from Jawaker; Pocket not explicit |
| Maximum Dash Calls | Two; no separate cap on ordinary zero estimates | Provisional |
| Auction turn order | Dealer's next seat; passing permanently exits | Provisional details |
| Estimate order | Caller’s next seat; Dash callers skipped | Provisional details |
| Fixed-round ties | First highest estimate is caller | Provisional |
| Super call | An 8+ auction before fixed-round estimates | Exact timing needs checking |
| All-pass round | Consumes one of the 18 rounds | Tutorial says next round; app counter needs checking |
| Missing suits | Inferred from legal play; no pre-auction Avoid announcement | Deliberate omission pending confirmation |

Comparison: https://blog.jawaker.com/en/estimation-rules-en/

`Rules` in `estimation/game.py` exposes miss_base, no_exact_doubles,
dash_replaces_base, max_dash_calls and max_risk. For a no-base-penalty variant,
use `Rules(miss_base=0)`. That alone does not make the entire profile identical
to Jawaker. Training/evaluation commands currently use the default profile.
Changing the scoring profile invalidates the strength claims for a checkpoint;
retrain and evaluate under the chosen profile.

An exact Pocket-compatible release needs in-app scoring examples for: ordinary
miss, failed Dash Call, sole Dash winner/loser, chained doubles, With, and Risk.
The engine tests validate the declared profile, not proprietary app parity.
