"""Rules engine. Seats increase in the physical counterclockwise direction.

The provisional scoring choices are explicit in Rules, not hidden in the bot.
Agents receive immutable observations, never a Game or opponents' hands.
"""
from dataclasses import dataclass, asdict
import random

SUITS = ('C', 'D', 'H', 'S', 'NT')
RANKS = '23456789TJQKA'

def card_name(c):
    return RANKS[c % 13] + SUITS[c // 13]

def strength(c, lead, trump):
    return (2 if c // 13 == trump else 1 if c // 13 == lead else 0) * 13 + c % 13

def winner(table, trump):
    lead = table[0][1] // 13
    return max(table, key=lambda pc: strength(pc[1], lead, trump))[0]

@dataclass(frozen=True)
class Rules:
    # Pocket's table says -10 for a failed round. Its "Tricks Amount" loss
    # is ambiguous; use absolute estimation error until checked in the app.
    miss_base: int = 10
    no_exact_doubles: bool = True
    dash_replaces_base: bool = True
    max_dash_calls: int = 2
    max_risk: int = 6

    def __post_init__(self):
        if not 0 <= self.max_dash_calls <= 2:
            raise ValueError('This profile supports zero to two Dash Calls')
        if self.miss_base < 0 or self.max_risk < 0:
            raise ValueError('Scoring penalties and risk cap must be nonnegative')

@dataclass(frozen=True)
class Observation:
    seat: int
    hand: tuple
    phase: str
    legal: tuple
    trump: int
    caller: int
    bids: tuple
    taken: tuple
    table: tuple
    played: tuple
    voids: tuple
    dash: tuple
    high_bid: tuple | None
    auction: tuple
    scores: tuple
    round_index: int
    multiplier: int
    risk_seat: int = -1

    def to_dict(self):
        return asdict(self)

class Game:
    def __init__(self, seed=0, round_index=0, dealer=0, scores=None,
                 multiplier=1, rules=None):
        self.rules = rules or Rules()
        deck = list(range(52))
        random.Random(seed).shuffle(deck)
        self.hands = [sorted(deck[i*13:(i+1)*13]) for i in range(4)]
        self.round_index, self.dealer = round_index, dealer
        self.scores = list(scores or [0]*4)
        self.multiplier = multiplier
        self.seat = (dealer + 1) % 4
        self.phase = 'dash'
        self.trump = 4 if round_index < 13 else 17-round_index
        self.caller = -1
        self.bids = [-1]*4
        self.taken = [0]*4
        self.table, self.played, self.auction = [], [], []
        self.voids = [set() for _ in range(4)]
        self.dash, self.passed = [False]*4, set()
        self.high_bid = None
        self.dash_count = 0
        self.queue = []
        self.risk_seat = -1
        self.result = None
        self.next_multiplier = 1

    def legal(self):
        if self.phase == 'done': return ()
        if self.phase == 'dash':
            return (0, 1) if sum(self.dash) < self.rules.max_dash_calls else (0,)
        if self.phase in ('auction', 'super'):
            low = 4 if self.phase == 'auction' else 8
            # Bid id 1 + (tricks-4)*5 + suit; 0 is pass.
            return (0,) + tuple(1+(n-4)*5+s for n in range(low,14) for s in range(5)
                if self.high_bid is None or (n,s) > self.high_bid)
        if self.phase == 'estimate':
            cap = self.high_bid[0] if self.high_bid else 13
            values = list(range(cap+1))
            if len(self.queue) == 1:
                forbidden = 13-sum(b for b in self.bids if b >= 0)
                values = [b for b in values if b != forbidden]
            return tuple(values)
        hand = self.hands[self.seat]
        if not self.table: return tuple(hand)
        follow = [c for c in hand if c//13 == self.table[0][1]//13]
        return tuple(follow or hand)

    def observe(self):
        return Observation(self.seat, tuple(self.hands[self.seat]), self.phase,
            self.legal(), self.trump, self.caller, tuple(self.bids),
            tuple(self.taken), tuple(self.table), tuple(self.played),
            tuple(tuple(sorted(v)) for v in self.voids), tuple(self.dash),
            self.high_bid, tuple(self.auction), tuple(self.scores),
            self.round_index, self.multiplier, self.risk_seat)

    def _begin_estimates(self):
        if self.high_bid:
            self.trump = self.high_bid[1]
            self.bids[self.caller] = self.high_bid[0]
            start = (self.caller+1)%4
        else:
            start = (self.dealer+1)%4
        self.queue = [(start+i)%4 for i in range(4)
                      if self.bids[(start+i)%4] < 0]
        if not self.queue: raise RuntimeError('No estimating players')
        self.phase, self.seat = 'estimate', self.queue[0]

    def step(self, action):
        if action not in self.legal():
            raise ValueError(f'Illegal {self.phase} action: {action}')
        p = self.seat
        if self.phase == 'dash':
            self.dash[p] = bool(action)
            if action:
                self.bids[p] = 0
                self.passed.add(p)
            self.dash_count += 1
            self.seat = (p+1)%4
            if self.dash_count == 4:
                self.phase = 'auction' if self.round_index < 13 else 'super'
                while self.seat in self.passed: self.seat = (self.seat+1)%4
            return
        if self.phase in ('auction', 'super'):
            self.auction.append((p, action))
            if action == 0: self.passed.add(p)
            else:
                self.high_bid = ((action-1)//5+4, (action-1)%5)
                self.caller = p
            active = [s for s in range(4) if s not in self.passed]
            if not active:
                if self.phase == 'super': self._begin_estimates()
                else:
                    self.phase, self.result = 'done', [0]*4
                    self.next_multiplier = self.multiplier*2
                return
            if len(active) == 1 and self.high_bid is not None:
                self._begin_estimates()
                return
            self.seat = (p+1)%4
            while self.seat in self.passed: self.seat = (self.seat+1)%4
            return
        if self.phase == 'estimate':
            self.bids[p] = action
            self.risk_seat = p
            self.queue.pop(0)
            if self.queue: self.seat = self.queue[0]
            else:
                if self.caller == -1:
                    order = [(self.dealer+1+i)%4 for i in range(4)]
                    self.caller = max(order, key=lambda s:self.bids[s])
                self.phase, self.seat = 'play', self.caller
            return
        if self.table and action//13 != self.table[0][1]//13:
            self.voids[p].add(self.table[0][1]//13)
        self.hands[p].remove(action)
        self.table.append((p,action))
        self.played.append((p,action))
        if len(self.table) == 4:
            self.seat = winner(self.table,self.trump)
            self.taken[self.seat] += 1
            self.table = []
            if sum(self.taken) == 13:
                self.result, self.next_multiplier = score(self.bids,self.taken,
                    self.caller,self.dash,self.risk_seat,self.multiplier,self.rules)
                self.phase = 'done'
        else: self.seat = (p+1)%4

def score(bids, taken, caller, dash, risk_seat, multiplier=1, rules=None):
    rules = rules or Rules()
    if (len(bids)!=4 or len(taken)!=4 or len(dash)!=4 or
        any(not isinstance(n,int) or n<0 or n>13 for n in list(bids)+list(taken)) or
        sum(taken)!=13 or sum(bids)==13 or caller not in range(4) or
        risk_seat not in range(4) or multiplier<1 or
        any(dash[p] and bids[p]!=0 for p in range(4))):
        raise ValueError('Invalid terminal deal for scoring')
    exact = [b==t for b,t in zip(bids,taken)]
    count = sum(exact)
    if count == 0 and rules.no_exact_doubles:
        return [0]*4, multiplier*2
    over = sum(bids)>13
    risk = min(rules.max_risk, abs(sum(bids)-13)//2)
    result = []
    for p in range(4):
        sign = 1 if exact[p] else -1
        base = 10+bids[p] if exact[p] else -rules.miss_base-abs(bids[p]-taken[p])
        if dash[p] and rules.dash_replaces_base:
            base = sign*(25 if over else 33)
        else:
            if p==caller or bids[p]==bids[caller]: base += sign*10
            if dash[p]: base += sign*(25 if over else 33)
        if (exact[p] and count==1) or (not exact[p] and count==3): base += sign*10
        if p==risk_seat: base += sign*10*risk
        result.append(base*multiplier)
    return result, 1

def run_deal(agents, seed=0, round_index=0, dealer=0, scores=None, multiplier=1, rules=None):
    g = Game(seed,round_index,dealer,scores,multiplier,rules)
    while g.phase != 'done': g.step(agents[g.seat].act(g.observe()))
    return g

def run_match(agents, seed=0, rules=None):
    totals, multiplier, rounds = [0]*4, 1, []
    rng = random.Random(seed)
    for r in range(18):
        g = run_deal(agents,rng.randrange(2**32),r,r%4,totals,multiplier,rules)
        totals = [a+b for a,b in zip(totals,g.result)]
        multiplier = g.next_multiplier
        rounds.append({'round':r+1,'bids':g.bids,'taken':g.taken,'score':g.result})
    return totals, rounds
