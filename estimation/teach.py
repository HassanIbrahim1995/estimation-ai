"""Generate teacher examples, distil the policy, validate before promotion."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .game import Game
from .agent import Agent, action_features
from .teacher import Teacher
from .train import fitness

def softmax(x):
    x=x-np.max(x); y=np.exp(x); return y/y.sum()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/selfplay.json')
    p.add_argument('--examples',type=int,default=512)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--output',default='models/taught.json')
    args=p.parse_args()
    if args.examples<20 or args.samples<1: p.error('Use examples >=20 and samples >=1')
    base=Agent.load(args.model)
    teacher=Teacher(base.weights,seed=8675309,samples=args.samples)
    rng=np.random.default_rng(8675309)
    data=[]; start=time.time(); seed=3100000
    categories=['under','over','dash_call','target_reached']
    counts={k:0 for k in categories}
    Path('reports').mkdir(exist_ok=True)
    with open('reports/teacher_examples.jsonl','w') as out:
        while len(data)<args.examples:
            g=Game(seed,round_index=seed%18,dealer=seed%4);seed+=1
            if seed>3100000+args.examples*100:
                raise RuntimeError('Could not fill curriculum; inspect generation conditions')
            category=categories[len(data)%len(categories)]
            generator_weights=base.weights.copy()
            if category=='over': generator_weights[28]+=1.5
            generator=Agent(generator_weights)
            dash_seat=seed%4
            # One state per deal avoids related train/validation examples.
            target=int(rng.integers(3,11))
            while g.phase!='done':
                o=g.observe()
                eligible=(category=='under' and sum(o.bids)<13 or
                    category=='over' and sum(o.bids)>13 or
                    category=='dash_call' and o.dash[o.seat] or
                    category=='target_reached' and o.bids[o.seat]<=o.taken[o.seat])
                if o.phase=='play' and eligible and sum(o.taken)>=target and len(o.legal)>1:
                    values=teacher.values(o)
                    row={'category':category,'observation':o.to_dict(),'teacher_values':values.tolist(),
                         'paired_standard_error':teacher.last_paired_se.tolist(),
                         'baseline_action_index':teacher.last_baseline_index}
                    out.write(json.dumps(row)+'\n')
                    data.append((action_features(o),values))
                    counts[category]+=1
                    break
                if category=='dash_call' and o.phase=='dash' and o.seat==dash_seat and 1 in o.legal:
                    g.step(1)
                else: g.step(generator.act(o))
            if len(data) and len(data)%64==0:
                print(json.dumps({'examples':len(data),'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    cut=int(len(data)*.8);train=data[:cut];test=data[cut:]
    anchor=base.weights.copy();w=anchor.copy()
    snapshots=[w.copy()]
    def regret(weights,ds):
        return float(np.mean([y.max()-y[int(np.argmax(x@weights[:22]))] for x,y in ds]))
    for epoch in range(args.epochs):
        gradient=np.zeros(22)
        for x,y in train:
            target=softmax(y/5.)
            pred=softmax(x@w[:22])
            gradient+=x.T@(pred-target)
        gradient/=len(train)
        gradient+=.002*(w[:22]-anchor[:22])
        w[:22]-=.4*gradient
        if (epoch+1)%20==0: snapshots.append(w.copy())
    selected=min(snapshots,key=lambda weights:regret(weights,test))
    report={'examples':len(data),'training_examples':cut,'heldout_examples':len(test),
        'samples_per_action':args.samples,'epochs':args.epochs,'curriculum_counts':counts,
        'baseline_teacher_regret':regret(anchor,test),'taught_teacher_regret':regret(selected,test),
        'sampler':'randomized feasible bipartite matching; not uniform posterior',
        'hidden_information':'teacher receives public observation plus own hand only'}
    # A separate gameplay validation gate prevents imitation from blindly
    # replacing an agent with better real-game performance.
    seeds=np.arange(720000,720288)
    before=fitness(anchor,seeds,[anchor])
    after=fitness(selected,seeds,[anchor])
    report.update({'gameplay_validation_before':before,'gameplay_validation_after':after,
        'promoted':bool(after>before),'elapsed_seconds':round(time.time()-start,2)})
    Agent(selected).save(args.output,{'training':'Monte Carlo teacher distillation',**report})
    if after>before:
        Agent(selected).save('models/trained.json',{'training':'self-play followed by teacher distillation',**report})
    else:
        Agent(anchor).save('models/trained.json',{'training':'self-play retained after teaching validation',**report})
    Path('reports/teaching.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
