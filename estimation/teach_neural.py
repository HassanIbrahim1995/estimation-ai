"""Distil uncertainty-screened search labels; validate without replacing a champion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .agent import Agent, action_features
from .game import Observation, run_deal
from .teach import softmax


def teacher_target(base, x, row):
    """SE is paired to the recorded teacher baseline, not an arbitrary model."""
    y = np.asarray(row['teacher_values'], dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError('Invalid teacher values')
    logits = base.action_values(x)
    prior = softmax(logits)
    initial = int(np.argmax(logits))
    se = np.asarray(row.get('paired_standard_error', [float('inf')]*len(y)))
    if se.shape != y.shape or np.isnan(se).any() or (se < 0).any():
        raise ValueError('Invalid paired standard errors')
    # Missing provenance is uncertainty, not zero uncertainty.
    if row.get('baseline_action_index') != initial:
        return prior, False
    advantage = y-y[initial]
    confident = np.flatnonzero(advantage > 1.96*se+1.)
    if not len(confident):
        return prior, False
    best = int(confident[np.argmax(advantage[confident])])
    target = .3*prior
    target[best] += .7
    return target, True


def paired_validation(base, candidate, seeds):
    """Four seats per seed; confidence interval uses independent seed blocks."""
    before, after = [], []
    for seed in seeds:
        b, a = [], []
        for seat in range(4):
            for bot, rewards in ((base, b), (candidate, a)):
                agents = [Agent(base.weights, neural=base.neural) for _ in range(4)]
                agents[seat] = bot
                g = run_deal(agents, int(seed), int(seed)%18, int(seed)%4)
                rewards.append(g.result[seat]-(sum(g.result)-g.result[seat])/3)
        before.append(np.mean(b)); after.append(np.mean(a))
    delta = np.asarray(after)-before
    error = 1.96*delta.std(ddof=1)/np.sqrt(len(delta))
    return {'gameplay_score_advantage_before': float(np.mean(before)),
            'gameplay_score_advantage_after': float(np.mean(after)),
            'paired_improvement': float(delta.mean()),
            'paired_improvement_approx_95pct_ci': [float(delta.mean()-error), float(delta.mean()+error)],
            'validation_seed_blocks': len(seeds), 'validation_deals_per_model': 4*len(seeds)}


def regret(bot, ds):
    return float(np.mean([y.max()-y[int(np.argmax(bot.action_values(x)))] for x,y,_,_ in ds]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/selfplay.json')
    p.add_argument('--data', default='reports/teacher_examples.jsonl')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seed', type=int, default=67221)
    p.add_argument('--validation-seed-start', type=int, default=740000)
    p.add_argument('--validation-seeds', type=int, default=144)
    p.add_argument('--output', default='models/neural_candidate.json')
    p.add_argument('--report', default='reports/neural_teaching.json')
    p.add_argument('--promote-to', help='Optional champion path; written ONLY if both gates pass')
    a = p.parse_args()
    if a.epochs < 1 or a.validation_seeds < 2:
        p.error('epochs >= 1 and validation-seeds >= 2 are required')
    if Path(a.output).resolve() == Path(a.model).resolve():
        p.error('candidate output must not overwrite the input checkpoint')
    base = Agent.load(a.model)
    data = []
    seen = set()
    deal_seeds = set()
    raw = Path(a.data).read_bytes()
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if 'deal_seed' in row:
            if row['deal_seed'] in deal_seeds:
                raise ValueError('Repeated deal_seed: use one independent position per deal')
            deal_seeds.add(row['deal_seed'])
        key = json.dumps(row['observation'], sort_keys=True)
        if key in seen:
            raise ValueError('Duplicate observation: would risk train/validation leakage')
        seen.add(key)
        o = Observation(**row['observation'])
        if o.phase != 'play' or len(o.legal) < 2:
            raise ValueError('Training requires play decisions with multiple legal actions')
        x = action_features(o); y = np.array(row['teacher_values'])
        target, confident = teacher_target(base, x, row)
        data.append((x, y, target, confident))
    if len(data) < 20:
        p.error('At least 20 independent examples are required')
    # Generator emits one position per deal in balanced, interleaved categories.
    # Keep the final chronological block out of gradient training.
    cut = int(len(data)*.8); training = data[:cut]; validation = data[cut:]
    rng = np.random.default_rng(a.seed)
    n = ({k:v.copy() for k,v in base.neural.items()} if base.neural is not None else
         {'w1':rng.normal(0,.15,(22,32)), 'b1':np.zeros(32), 'w2':np.zeros(32)})
    m = {k:np.zeros_like(v) for k,v in n.items()}; v = {k:np.zeros_like(v) for k,v in n.items()}
    baseline_regret = regret(base, validation)
    snapshots = [(baseline_regret, 0, base)]; step = 0
    for epoch in range(a.epochs):
        indices = rng.permutation(len(training))
        for offset in range(0,len(indices),32):
            grad = {k:np.zeros_like(v) for k,v in n.items()}
            batch = indices[offset:offset+32]
            for i in batch:
                x,y,target,_ = training[i]; h = np.tanh(x@n['w1']+n['b1'])
                logits = x@base.weights[:22]+h@n['w2']
                error = softmax(logits)-target
                grad['w2'] += h.T@error
                dh = (error[:,None]*n['w2'][None,:])*(1-h*h)
                grad['w1'] += x.T@dh; grad['b1'] += dh.sum(axis=0)
            step += 1
            for k in n:
                g = grad[k]/len(batch)+.001*n[k]
                m[k] = .9*m[k]+.1*g; v[k] = .999*v[k]+.001*g*g
                n[k] -= .003*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if epoch+1 in (1,5,10,a.epochs) or (epoch+1)%20 == 0:
            bot = Agent(base.weights, neural={k:v.copy() for k,v in n.items()})
            snapshots.append((regret(bot,validation),epoch+1,bot))
    best_regret, epoch, chosen = min(snapshots,key=lambda item:item[0])
    seeds = range(a.validation_seed_start,a.validation_seed_start+a.validation_seeds)
    gate = paired_validation(base, chosen, seeds)
    promoted = best_regret < baseline_regret and gate['paired_improvement_approx_95pct_ci'][0] > 0
    report = {'architecture':'22 features -> 32 tanh units -> scalar residual action score',
        'neural_parameters':768, 'training_examples':cut, 'heldout_examples':len(validation),
        'epochs':a.epochs, 'selected_epoch':epoch, 'teacher_regret_before':baseline_regret,
        'teacher_regret_after':best_regret, 'promoted':bool(promoted), 'training_seed':a.seed,
        'validation_seed_start':a.validation_seed_start,
        'data_sha256':hashlib.sha256(raw).hexdigest(),
        'confident_teacher_corrections':sum(row[3] for row in data),
        'confidence_note':'Normal approximation over paired seed blocks; selection set, not final test.', **gate}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    chosen.save(a.output, report)
    if promoted and a.promote_to:
        Path(a.promote_to).parent.mkdir(parents=True,exist_ok=True)
        chosen.save(a.promote_to,report)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
