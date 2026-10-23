"""Compact learned action scorer. All features use public information only."""
import json
import math
import random
from pathlib import Path
import numpy as np
from .game import strength, winner

FEATURES = [
    'win_probability_when_need', 'win_probability_when_full',
    'rank_when_need', 'rank_when_full', 'trump_when_need', 'trump_when_full',
    'master_when_need', 'master_when_full', 'long_suit', 'short_suit_when_full',
    'last_to_play_win_when_need', 'last_to_play_win_when_full',
    'win_times_required_fraction', 'win_when_over', 'win_when_under',
    'deny_current_winner_when_they_need_one', 'dump_on_full_opponent',
    'rank_if_losing_when_full', 'rank_if_losing_when_need',
    'win_times_remaining_tricks', 'lead_master_when_need', 'lead_low_when_full',
    'bid_ace', 'bid_king', 'bid_queen', 'bid_jack', 'bid_trump_length',
    'bid_void', 'bid_bias', 'auction_caution', 'dash_threshold'
]

DEFAULT = np.array([
    5., -9., -.7, 1., -.5, .2, 1., -1., -.1, .2,
    3., -5., 2., .3, -.3, .2, .2, 1.5, .2, 0., .5, .5,
    1., .75, .5, .3, .6, .15, .3, -.25, -.4
], dtype=float)

def estimate(hand, trump, w):
    lengths = [sum(c//13==s for c in hand) for s in range(4)]
    counts = [0.]*4
    for c in hand:
        r, s = c%13, c//13
        if r >= 9:
            protection = min(1.,lengths[s]/(13-r))
            counts[12-r] += protection
    value = sum(counts[i]*w[22+i] for i in range(4)) + w[28]
    if trump < 4:
        value += max(0,lengths[trump]-3)*w[26]
        value += sum(n==0 for n in lengths)*min(lengths[trump],3)*w[27]
    return max(0.,min(13.,value))

def action_features(o):
    hand = set(o.hand)
    seen = {c for _,c in o.played}
    unseen = [c for c in range(52) if c not in hand and c not in seen]
    unseen_suits = [[c for c in unseen if c//13==s] for s in range(4)]
    lengths = [sum(c//13==s for c in o.hand) for s in range(4)]
    remaining = 13-sum(o.taken)
    need = o.bids[o.seat]-o.taken[o.seat]
    wants = float(need>0)
    full = 1.-wants
    current = winner(o.table,o.trump) if o.table else -1
    later = [(o.seat+i)%4 for i in range(1,4-len(o.table))]
    rows = []
    for c in o.legal:
        suit,rank = c//13,c%13/12
        lead = o.table[0][1]//13 if o.table else suit
        candidate_wins = not o.table or strength(c,lead,o.trump)>max(
            strength(t,lead,o.trump) for _,t in o.table)
        probability = float(candidate_wins)
        for p in later:
            pool = [x for s in range(4) if s not in o.voids[p] for x in unseen_suits[s]]
            if not pool: continue
            # Approximate chances of a higher legal card, not knowledge of a hand.
            if lead in o.voids[p]:
                beats = sum(strength(x,lead,o.trump)>strength(c,lead,o.trump) for x in pool)
            else:
                beats = sum(x//13==lead and strength(x,lead,o.trump)>strength(c,lead,o.trump) for x in pool)
                if suit != o.trump and lead != o.trump and o.trump < 4:
                    # Approximate chance of being void and ruffing.
                    no_follow = (1-len(unseen_suits[lead])/max(1,len(pool)))**remaining
                    beats += no_follow*len(unseen_suits[o.trump])
            probability *= max(0.,1-beats/len(pool))**remaining
        master = float(not any(x>c for x in unseen_suits[suit]))
        last = float(len(o.table)==3)
        over = float(sum(o.bids)>13)
        deny = float(current>=0 and o.bids[current]-o.taken[current]==1)
        dump = float(current>=0 and o.taken[current]>=o.bids[current])
        rows.append([
            probability*wants,probability*full,rank*wants,rank*full,
            (suit==o.trump)*wants,(suit==o.trump)*full,master*wants,master*full,
            lengths[suit]/13,full/lengths[suit],last*probability*wants,
            last*probability*full,probability*max(0,need)/remaining,
            probability*over*wants,probability*(1-over)*wants,
            probability*deny, (1-probability)*dump,rank*(1-probability)*full,
            rank*(1-probability)*wants,probability*remaining/13,
            (not o.table)*master*wants,(not o.table)*(1-rank)*full
        ])
    return np.array(rows,dtype=float)

class Agent:
    def __init__(self, weights=None, seed=0, random_play=False, epsilon=0., neural=None):
        self.weights = np.array(DEFAULT if weights is None else weights,dtype=float)
        if self.weights.shape != DEFAULT.shape or not np.isfinite(self.weights).all():
            raise ValueError('Invalid model weights')
        self.rng = random.Random(seed)
        self.random_play,self.epsilon = random_play,epsilon
        self.neural = None if neural is None else {k:np.array(v,dtype=float) for k,v in neural.items()}
        if self.neural is not None:
            n=self.neural
            if set(n)!={'w1','b1','w2'} or n['w1'].shape!=(22,32) or n['b1'].shape!=(32,) or n['w2'].shape!=(32,):
                raise ValueError('Invalid neural layer shapes')
            if not all(np.isfinite(v).all() for v in n.values()): raise ValueError('Nonfinite neural weights')

    def action_values(self,x):
        values=x@self.weights[:22]
        if self.neural is not None:
            n=self.neural
            values+=np.tanh(x@n['w1']+n['b1'])@n['w2']
        return values

    def act(self,o):
        if not o.legal: raise ValueError('No legal action in this observation')
        if o.phase == 'dash':
            # Zero before knowing trump requires a very weak hand in every suit.
            weak = max(estimate(o.hand,t,self.weights) for t in range(5))
            return int(1 in o.legal and weak < self.weights[30])
        if o.phase in ('auction','super'):
            choices = []
            for a in o.legal:
                if not a: continue
                n,t = (a-1)//5+4,(a-1)%5
                expected = estimate(o.hand,t,self.weights)
                if n <= expected-self.weights[29]:
                    choices.append((expected-.15*n,a))
            return max(choices)[1] if choices else 0
        if o.phase == 'estimate':
            expected = estimate(o.hand,o.trump,self.weights)
            return min(o.legal,key=lambda b:(abs(b-expected),b))
        if len(o.legal)==1: return o.legal[0]
        if self.random_play or self.rng.random()<self.epsilon:
            return self.rng.choice(o.legal)
        values = self.action_values(action_features(o))
        return o.legal[int(np.argmax(values))]

    def save(self,path,metadata=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({'format':2 if self.neural is not None else 1,'features':FEATURES,
            'weights':self.weights.tolist(),'neural':None if self.neural is None else
            {k:v.tolist() for k,v in self.neural.items()},'metadata':metadata or {}},indent=2))

    @classmethod
    def load(cls,path,**kwargs):
        data = json.loads(Path(path).read_text())
        if data.get('format') not in (1,2) or data.get('features')!=FEATURES:
            raise ValueError('Incompatible model format')
        return cls(data['weights'],neural=data.get('neural'),**kwargs)
