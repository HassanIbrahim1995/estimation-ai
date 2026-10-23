"""Generate balanced teacher examples without modifying saved models."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .game import Game
from .agent import Agent, action_features
from .teacher import Teacher

def softmax(x):
    x=x-np.max(x); y=np.exp(x); return y/y.sum()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/selfplay.json')
    p.add_argument('--examples',type=int,default=512)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--output',default='reports/curriculum_v2.jsonl')
    p.add_argument('--seed',type=int,default=4200000)
    args=p.parse_args()
    if args.examples<20 or args.samples<2: p.error('Use examples >=20 and samples >=2')
    base=Agent.load(args.model)
    if base.neural is not None: p.error('Curriculum currently requires a linear base model')
    teacher=Teacher(base.weights,seed=args.seed+100000,samples=args.samples)
    rng=np.random.default_rng(args.seed)
    data=[]; start=time.time(); seed=args.seed
    categories=['under','over','dash_call','target_reached']
    counts={k:0 for k in categories}
    Path('reports').mkdir(exist_ok=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w') as out:
        while len(data)<args.examples:
            g=Game(seed,round_index=seed%18,dealer=seed%4);seed+=1
            if seed>args.seed+args.examples*100:
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
                    category=='target_reached' and o.bids[o.seat]==o.taken[o.seat])
                if o.phase=='play' and eligible and sum(o.taken)>=target and len(o.legal)>1:
                    values=teacher.values(o)
                    row={'category':category,'deal_seed':seed-1,'samples':args.samples,'rules_profile':'pocket-inspired-provisional-v1','observation':o.to_dict(),'teacher_values':values.tolist(),
                         'paired_standard_error':teacher.last_paired_se.tolist(),
                         'baseline_action_index':teacher.last_baseline_index}
                    out.write(json.dumps(row)+'\n');out.flush()
                    data.append((action_features(o),values))
                    counts[category]+=1
                    break
                if category=='dash_call' and o.phase=='dash' and o.seat==dash_seat and 1 in o.legal:
                    g.step(1)
                else: g.step(generator.act(o))
            if len(data) and len(data)%64==0:
                print(json.dumps({'examples':len(data),'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    print(json.dumps({'examples':len(data),'counts':counts,'seconds':time.time()-start}),flush=True)

if __name__=='__main__': main()
