"""Seat-balanced held-out 18-round matches, with seed-block uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from .agent import Agent, DEFAULT
from .game import run_match

def benchmark(weights,seeds,random_opponents=False):
    blocks=[]; wins=[]; exact=[]; elapsed_rounds=0; skipped=0
    for seed in seeds:
        advantages=[]; seed_wins=[]; seed_exact=[]
        for seat in range(4):
            agents=[Agent(seed=int(seed)+j,random_play=random_opponents) for j in range(4)]
            agents[seat]=weights if isinstance(weights,Agent) else Agent(weights)
            scores,rounds=run_match(agents,int(seed))
            advantages.append(scores[seat]-(sum(scores)-scores[seat])/3)
            seed_wins.append((1/scores.count(max(scores))) if scores[seat]==max(scores) else 0.)
            played=[r for r in rounds if sum(r['taken'])==13]
            skipped+=len(rounds)-len(played)
            seed_exact.extend(r['bids'][seat]==r['taken'][seat] for r in played)
            elapsed_rounds+=len(rounds)
        blocks.append(float(np.mean(advantages)))
        wins.append(float(np.mean(seed_wins)))
        exact.extend(seed_exact)
    blocks=np.array(blocks)
    win_error=1.96*np.std(wins,ddof=1)/np.sqrt(len(wins)) if len(wins)>1 else None
    error=1.96*blocks.std(ddof=1)/np.sqrt(len(blocks)) if len(blocks)>1 else None
    return {'seed_blocks':len(seeds),'matches':len(seeds)*4,'rounds':elapsed_rounds,
        'mean_match_score_advantage':float(blocks.mean()),
        'approx_95pct_ci':None if error is None else [float(blocks.mean()-error),float(blocks.mean()+error)],
        'match_win_rate_ties_split':float(np.mean(wins)),
        'win_rate_approx_95pct_ci': None if win_error is None else
            [max(0.,float(np.mean(wins)-win_error)),min(1.,float(np.mean(wins)+win_error))],
        'all_pass_round_fraction':skipped/elapsed_rounds,
        'exact_bid_rate_on_played_deals':float(np.mean(exact)),
        'score_advantage_by_seed':blocks.tolist(), 'win_rate_by_seed':wins}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/trained.json')
    p.add_argument('--seeds',type=int,default=40)
    p.add_argument('--seed-start',type=int,default=900000)
    p.add_argument('--output',default='reports/evaluation.json')
    a=p.parse_args()
    if a.seeds<2: p.error('Use at least two seed blocks')
    seeds=list(range(a.seed_start,a.seed_start+a.seeds))
    model=Agent.load(a.model)
    result={'seed_start':a.seed_start,'rules':'pocket-inspired-provisional-v1',
        'notes':'Final seeds are disjoint from training and selection. CI clusters four seats by deal seed. No human tests.',
        'trained_vs_heuristic':benchmark(model,seeds),
        'untrained_vs_heuristic':benchmark(DEFAULT,seeds),
        'trained_vs_random_card_play':benchmark(model,seeds,True)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:({x:y for x,y in v.items() if not x.endswith('_by_seed')} if isinstance(v,dict) else v) for k,v in result.items()},indent=2))

if __name__=='__main__': main()
