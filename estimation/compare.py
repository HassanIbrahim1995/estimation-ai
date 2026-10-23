"""One-shot final comparison; never used to select a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from .agent import Agent
from .evaluate import benchmark

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',default='models/neural_v2.json')
    p.add_argument('--base',default='models/selfplay.json')
    p.add_argument('--seed-start',type=int,default=920000)
    p.add_argument('--seeds',type=int,default=60)
    p.add_argument('--output',default='reports/comparison_v2.json')
    a=p.parse_args()
    if a.seeds<2: p.error('At least two seed blocks required')
    seeds=list(range(a.seed_start,a.seed_start+a.seeds))
    report={'seed_start':a.seed_start,'rules':'pocket-inspired-provisional-v1',
            'opponents':'three frozen heuristic agents',
            'purpose':'final reporting only; not checkpoint selection'}
    for label,path in [('base',a.base),('candidate',a.candidate)]:
        report[label]=benchmark(Agent.load(path),seeds)
        print(label+' complete',flush=True)
    d=np.array(report['candidate']['score_advantage_by_seed'])-report['base']['score_advantage_by_seed']
    error=1.96*d.std(ddof=1)/np.sqrt(len(d))
    report['paired_match_score_delta']=float(d.mean())
    report['paired_match_score_delta_approx_95pct_ci']=[float(d.mean()-error),float(d.mean()+error)]
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:({x:y for x,y in v.items() if not x.endswith('_by_seed')} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))

if __name__=='__main__': main()
