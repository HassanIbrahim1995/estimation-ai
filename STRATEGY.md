# What we teach the player

The objective is an exact trick target. Maximizing raw tricks is usually wrong.
These are engineering hypotheses derived from the rules, not quotations from
an expert coaching manual or claims of optimal strategy.

| Lesson | Information to use | Training treatment |
|---|---|---|
| Estimate realistically | High cards, suit length, trump, protected honors | Learned hand-evaluation and auction parameters |
| Change plans after reaching the target | Target minus tricks already taken | Separate action features and a dedicated curriculum category |
| Over rounds | Total demand exceeds available tricks | Dedicated over-round examples with legal aggressive bids |
| Under rounds | Some player must take unwanted tricks | Dedicated under-round examples |
| Protect zero | Winning any trick breaks the contract | Forced legal Dash Call scenarios; evaluate ways to escape high cards |
| Remember cards | Played cards and unseen higher ranks | Public card-memory features |
| Remember voids | A player legally fails to follow a suit | Exclude that suit from their sampled future hand |
| Consider opponents | Other targets and current trick winner | Features plus full-round score comparison in rollouts |
| Think beyond one trick | Winning now changes who leads next | Teacher simulates the rest of the deal |

For example, a high card can be an asset while you still need tricks and a
liability after you have reached your target. A low trump may win now but force
you to lead into another unwanted win later. A teacher label therefore comes
from the whole simulated continuation, not a fixed “always play high/low” rule.

## The teaching pipeline

1. A full rules engine generates legal positions.
2. The curriculum has equal counts of under, over, Dash Call and target-reached
   positions. Aggressive legal bidding generates over situations; forced legal
   Dash declarations provide rare zero-target practice. These situations need
   not represent optimal bidding.
3. For each position, the teacher receives only the actor's hand and public
   information. A randomized bipartite matching assigns unseen cards to
   opponents, preserving remaining hand sizes and revealed voids.
4. All legal root actions are tested on the same sampled worlds. Subsequent
   players see only their own hand and public history during each rollout.
5. Terminal score advantage supplies action-quality labels. A softmax loss
   distils these labels into the compact model. Bidding parameters are frozen
   during this teaching stage.
6. We select an imitation checkpoint using disjoint held-out positions, then
   gate promotion using a separate gameplay validation set. Final match seeds
   are separate again.

The default teacher objective is round score minus the mean opponents' round
score. This is not a perfect proxy for the chance of winning a whole match.
The sampler is **not** a uniform or calibrated posterior and does not yet use
bidding likelihoods. Rollouts inherit the baseline's errors. A finite sample
can recommend a poor move; teacher labels are estimates, not ground truth.

The first unconstrained teaching set contained only under rounds. It was kept
as a diagnostic, then replaced by the balanced curriculum. This illustrates
why self-play alone is insufficient: it can miss entire kinds of decisions.

## Provenance and limits

Pocket's English and Arabic tutorials establish the core mechanics:
https://blog.pocketestimation.com/en-how-to-play-estimation/
https://blog.pocketestimation.com/how-to-play-estimation/

Egyptian federation rules differ in auction and all-pass handling. They are
comparison material, not silently merged into the selected app-style profile:
https://egyestimation.org/rules/03.htm

Research on hidden-information search motivates considering information sets
rather than pretending all cards are visible. This implementation is root
Monte Carlo sampling, **not** a reproduction of ISMCTS:
https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/

We have not played against Pocket's proprietary AI, collected human games,
established professional-level strength. A small residual neural candidate has
been trained; see RESULTS.md for its validation and promotion decision.
The remaining rule uncertainties in RULES.md still apply.
