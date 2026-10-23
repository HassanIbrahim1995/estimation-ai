"""Cross-entropy policy search against a league of frozen and historical bots.

No labeled human games or hidden cards are supplied to policies. Optimization
uses terminal score advantage; this is evolutionary self-play, not deep RL.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .agent import Agent, DEFAULT
from .game import run_deal

def fitness(w,seeds,opponents):
    agent = w if isinstance(w,Agent) else Agent(w)
    rewards = []
    for i,seed in enumerate(seeds):
        seat = i%4
        agents = [Agent(opponents[(i+j)%len(opponents)]) for j in range(4)]
        agents[seat] = agent
        g = run_deal(agents,int(seed),i%18,i%4)
        rewards.append(g.result[seat]-(sum(g.result)-g.result[seat])/3)
    return float(np.mean(rewards))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generations',type=int,default=30)
    p.add_argument('--population',type=int,default=12)
    p.add_argument('--deals',type=int,default=36)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--resume')
    p.add_argument('--output',default='models/trained.json')
    args=p.parse_args()
    if args.generations<1 or args.population<4 or args.deals<4: p.error('Use positive generations, population >=4, deals >=4')
    rng=np.random.default_rng(args.seed)
    mean=Agent.load(args.resume).weights.copy() if args.resume else DEFAULT.copy()
    sigma=np.array([.8]*22+[.1,.12,.1,.08,.12,.08,.2,.15,.25])
    champion=mean.copy()
    league=[DEFAULT.copy(),champion.copy()]
    # Validation seeds are fixed; final benchmark uses disjoint seeds.
    validation=np.arange(700000,700144)
    best=fitness(champion,validation,[DEFAULT])
    history=[]
    start=time.time()
    total=len(validation)
    for generation in range(args.generations):
        seeds=rng.integers(0,600000,size=args.deals)
        candidates=mean+rng.normal(size=(args.population,len(mean)))*sigma
        candidates[0]=champion
        # Keep the hand estimator and auction parameters in sensible ranges.
        candidates[:,22:28]=np.clip(candidates[:,22:28],0.,1.8)
        candidates[:,29]=np.clip(candidates[:,29],-.5,2.)
        candidates[:,30]=np.clip(candidates[:,30],-.5,1.)
        values=[fitness(w,seeds,league) for w in candidates]
        total+=args.population*args.deals
        elite=candidates[np.argsort(values)[-max(2,args.population//4):]]
        mean=.4*mean+.6*elite.mean(axis=0)
        sigma=np.maximum(.06,.75*sigma+.25*elite.std(axis=0))
        contender=candidates[int(np.argmax(values))]
        val=fitness(contender,validation,[DEFAULT])
        total+=len(validation)
        accepted=val>best
        if accepted:
            champion=contender.copy(); best=val
            league.append(champion.copy())
            if len(league)>6: league.pop(1)
        row={'generation':generation+1,'training_fitness':max(values),
             'validation_fitness':val,'best_validation':best,'accepted':accepted,
             'simulated_deals':total,'elapsed_seconds':round(time.time()-start,2)}
        history.append(row)
        Agent(champion).save(args.output,{'algorithm':'cross-entropy evolutionary league self-play',
            'seed':args.seed,'simulated_deals':total,'generations':generation+1,
            'validation_score_advantage':best,'status':'research prototype; no human validation',
            'rules':'pocket-inspired-provisional-v1'})
        Path('reports').mkdir(exist_ok=True)
        Path('reports/training.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)

if __name__=='__main__': main()
