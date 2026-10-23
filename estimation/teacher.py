"""Public-information Monte Carlo teacher; not an omniscient solver or ISMCTS.

Sample hidden hands consistent with remaining card counts and revealed voids.
Each rollout agent gets only its own observation. Evaluate all root actions on
the same worlds to reduce noise. Assignment sampling is feasible, not uniform;
bidding likelihoods are not modelled yet.
"""
import random
import numpy as np
from .game import Game
from .agent import Agent

def sample_hands(o,rng):
    known=set(o.hand)|{c for _,c in o.played}
    unseen=[c for c in range(52) if c not in known]
    played_counts=[sum(p==s for p,_ in o.played) for s in range(4)]
    slots=[s for s in range(4) if s!=o.seat for _ in range(13-played_counts[s])]
    if len(unseen)!=len(slots): raise ValueError('Inconsistent public card counts')
    rng.shuffle(unseen); rng.shuffle(slots)
    # Bipartite matching: cards -> remaining hand slots respecting void suits.
    assigned={}
    options={c:[i for i,s in enumerate(slots) if c//13 not in o.voids[s]] for c in unseen}
    for opts in options.values(): rng.shuffle(opts)
    def assign(c,visited):
        for i in options[c]:
            if i in visited: continue
            visited.add(i)
            if i not in assigned or assign(assigned[i],visited):
                assigned[i]=c; return True
        return False
    for c in sorted(unseen,key=lambda c:len(options[c])):
        if not assign(c,set()): raise ValueError('Public voids cannot describe a valid deal')
    hands=[[] for _ in range(4)]
    hands[o.seat]=list(o.hand)
    for i,c in assigned.items(): hands[slots[i]].append(c)
    return [sorted(h) for h in hands]

def reconstruct(o,hands):
    if o.phase!='play' or o.risk_seat not in range(4):
        raise ValueError('Teacher requires a play observation with risk_seat')
    g=Game(0,o.round_index,scores=o.scores,multiplier=o.multiplier)
    g.phase='play';g.seat=o.seat;g.hands=[h.copy() for h in hands]
    g.trump=o.trump;g.caller=o.caller;g.bids=list(o.bids);g.taken=list(o.taken)
    g.table=[tuple(x) for x in o.table];g.played=[tuple(x) for x in o.played]
    g.voids=[set(v) for v in o.voids];g.dash=list(o.dash)
    g.high_bid=o.high_bid;g.auction=list(o.auction);g.risk_seat=o.risk_seat
    return g

class Teacher(Agent):
    def __init__(self,weights=None,seed=0,samples=12,max_remaining=13):
        super().__init__(weights,seed)
        if samples<1: raise ValueError('At least one sample is required')
        self.samples=samples;self.max_remaining=max_remaining
        self.rollout=Agent(self.weights)

    def values(self,o):
        returns=np.zeros((self.samples,len(o.legal)))
        for sample in range(self.samples):
            hands=sample_hands(o,self.rng)
            for i,a in enumerate(o.legal):
                g=reconstruct(o,hands);g.step(a)
                while g.phase!='done':
                    g.step(self.rollout.act(g.observe()))
                returns[sample,i]=(g.result[o.seat]-(sum(g.result)-g.result[o.seat])/3)/o.multiplier
        # Normalize chained score multipliers for labels; action ordering unchanged.
        baseline=o.legal.index(super().act(o))
        paired=returns-returns[:,baseline,None]
        self.last_paired_se=paired.std(axis=0,ddof=1)/np.sqrt(self.samples) if self.samples>1 else np.full(len(o.legal),float('inf'))
        self.last_baseline_index=baseline
        return returns.mean(axis=0)

    def act(self,o):
        if o.phase!='play' or len(o.legal)==1 or len(o.hand)>self.max_remaining:
            return super().act(o)
        values=self.values(o)
        best=np.flatnonzero(np.isclose(values,values.max()))
        fallback=super().act(o)
        if fallback in [o.legal[i] for i in best]: return fallback
        return o.legal[int(best[0])]
