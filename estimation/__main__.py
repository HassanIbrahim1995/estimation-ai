import argparse
import json
import sys
from .agent import Agent
from .game import Game, Observation, card_name, SUITS, run_match

def main():
    p=argparse.ArgumentParser(description='Local Estimation bot')
    p.add_argument('mode',choices=['play','demo','predict'])
    p.add_argument('--model',default='models/trained.json')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--search-samples',type=int,default=0,
        help='Optional Monte Carlo lookahead; slower, experimental')
    a=p.parse_args()
    bots=[Agent.load(a.model,seed=a.seed+i) for i in range(4)]
    if a.search_samples:
        from .teacher import Teacher
        bots=[Teacher(b.weights,seed=a.seed+i,samples=a.search_samples,max_remaining=6)
              for i,b in enumerate(bots)]
    if a.mode=='predict':
        # JSON-lines integration: input is one public Observation per line.
        for line in sys.stdin:
            try:
                o=Observation(**json.loads(line))
                print(json.dumps({'action':bots[0].act(o)}),flush=True)
            except (ValueError,TypeError,KeyError) as e:
                print(json.dumps({'error':str(e)}),flush=True)
        return
    if a.mode=='demo':
        scores,rounds=run_match(bots,a.seed)
        print(json.dumps({'scores':scores,'rounds':rounds},indent=2)); return
    totals=[0]*4; multiplier=1
    print('You are seat 0. C=clubs D=diamonds H=hearts S=spades NT=no trump.')
    print('Research rule profile: see RULES.md before comparing with Pocket Estimation.')
    for r in range(18):
        g=Game(a.seed+r,r,r%4,totals,multiplier)
        print(f'\nRound {r+1}; total scores {totals}; multiplier {multiplier}')
        while g.phase!='done':
            o=g.observe()
            if o.seat==0:
                print('\nYour hand:', ' '.join(card_name(c) for c in o.hand))
                print('Phase:',o.phase,'Trump:',SUITS[o.trump], 'Bids:',o.bids,'Taken:',o.taken)
                print('Table:',[(s,card_name(c)) for s,c in o.table])
                def label(x):
                    if o.phase=='play': return card_name(x)
                    if o.phase in ('auction','super'): return 'pass' if x==0 else f'{(x-1)//5+4} {SUITS[(x-1)%5]}'
                    if o.phase=='dash': return 'Dash Call' if x else 'no Dash Call'
                    return str(x)
                print(' | '.join(f'{i}: {label(x)}' for i,x in enumerate(o.legal)))
                while True:
                    try:
                        choice=int(input('Choose option number: '))
                        if choice<0 or choice>=len(o.legal): raise ValueError()
                        action=o.legal[choice]; break
                    except (ValueError,IndexError): print('Enter one of the option numbers.')
            else: action=bots[o.seat].act(o)
            if o.phase=='play': print(f'Player {o.seat}: {card_name(action)}')
            g.step(action)
        totals=[x+y for x,y in zip(totals,g.result)]
        multiplier=g.next_multiplier
        print('Bids:',g.bids,'Taken:',g.taken,'Round scores:',g.result)
    print('Final scores:',totals)

if __name__=='__main__': main()
