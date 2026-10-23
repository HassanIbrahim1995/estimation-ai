import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from estimation.game import Game, Rules, score, winner, run_deal, run_match
from estimation.agent import Agent, action_features

class RulesTests(unittest.TestCase):
    def test_deal_is_full_unique_deck(self):
        for seed in range(100):
            g=Game(seed)
            self.assertEqual(sorted(sum(g.hands,[])),list(range(52)))
            self.assertEqual([len(h) for h in g.hands],[13]*4)

    def test_follow_suit_is_mandatory(self):
        g=Game(); g.phase='play'; g.seat=0
        g.hands[0]=[0,12,13,51]; g.table=[(3,5)]
        self.assertEqual(g.legal(),(0,12))
        with self.assertRaises(ValueError): g.step(51)

    def test_trump_and_no_trump_winners(self):
        table=[(0,12),(1,13),(2,11),(3,51)]
        self.assertEqual(winner(table,1),1)
        self.assertEqual(winner(table,4),0)
        self.assertEqual(winner(table,3),3)

    def test_all_pass_carries_multiplier(self):
        g=Game(multiplier=2)
        for _ in range(8): g.step(0)
        self.assertEqual(g.phase,'done')
        self.assertEqual(g.result,[0]*4)
        self.assertEqual(g.next_multiplier,4)

    def test_fixed_rounds_and_super_override(self):
        for r,t in zip(range(13,18),[4,3,2,1,0]):
            g=Game(round_index=r)
            for _ in range(8): g.step(0)
            self.assertEqual(g.phase,'estimate')
            self.assertEqual(g.trump,t)
        g=Game(round_index=13)
        for _ in range(4): g.step(0)
        g.step(1+(8-4)*5+2)
        for _ in range(3): g.step(0)
        self.assertEqual(g.trump,2)
        self.assertEqual(g.bids[g.caller],8)

    def test_dash_limit_and_zero_estimate(self):
        g=Game()
        g.step(1); g.step(1)
        self.assertEqual(g.legal(),(0,))
        g.step(0); g.step(0)
        self.assertEqual(sum(g.dash),2)
        for i in range(4):
            if g.dash[i]: self.assertEqual(g.bids[i],0)

    def test_last_estimate_cannot_sum_to_13(self):
        g=Game()
        for _ in range(4): g.step(0)
        g.step(1+(5-4)*5+4)
        for _ in range(3): g.step(0)
        g.step(3);g.step(3)
        self.assertNotIn(2,g.legal())

    def test_score_and_double(self):
        values,m=score([4,3,3,2],[4,3,3,3],0,[False]*4,3)
        self.assertEqual(values,[24,13,13,-21])
        self.assertEqual(m,1)
        self.assertEqual(score([4,3,3,2],[5,2,4,2],0,[False]*4,3)[0][3],22)
        self.assertEqual(score([4,3,3,2],[5,4,2,2],0,[False]*4,3,2)[0][3],44)
        self.assertEqual(score([4,3,3,2],[5,4,4,0],0,[False]*4,3,2),([0]*4,4))
        self.assertEqual(score([4,3,3,2],[4,3,3,3],0,[False]*4,3,
            rules=Rules(miss_base=0))[0][3],-11)

    def test_dash_and_risk_scoring(self):
        values,_=score([5,4,3,0],[5,4,4,0],0,[False,False,False,True],2)
        self.assertEqual(values[3],33)
        values,_=score([6,5,4,0],[6,3,4,0],0,[False,False,False,True],2)
        self.assertEqual(values[3],25)
        self.assertEqual(values[2],24)

    def test_invalid_terminal_scores_rejected(self):
        with self.assertRaises(ValueError):
            score([5,4,4,0],[5,4,4,0],0,[False]*4,3)
        with self.assertRaises(ValueError):
            score([4,3,3,2],[5,4,4,1],0,[False]*4,3)

    def test_observation_does_not_expose_other_hands(self):
        g=Game(seed=73,round_index=13)
        bots=[Agent() for _ in range(4)]
        while g.phase not in ('play','done'): g.step(bots[g.seat].act(g.observe()))
        self.assertEqual(g.phase,'play')
        before=g.observe(); decision=bots[g.seat].act(before)
        others=[s for s in range(4) if s!=g.seat]
        g.hands[others[0]],g.hands[others[1]]=g.hands[others[1]],g.hands[others[0]]
        after=g.observe()
        self.assertEqual(before,after)
        self.assertEqual(decision,bots[g.seat].act(after))
        self.assertNotIn('hands',after.to_dict())

    def test_many_games_preserve_cards_and_legality(self):
        for seed in range(60):
            g=Game(seed,seed%18)
            bots=[Agent(seed=i,epsilon=.1) for i in range(4)]
            steps=0
            while g.phase!='done':
                o=g.observe(); a=bots[g.seat].act(o)
                self.assertIn(a,g.legal())
                g.step(a); steps+=1
                self.assertLess(steps,300)
            if g.played:
                self.assertEqual(sum(g.taken),13)
                self.assertEqual(len({c for _,c in g.played}),52)
                self.assertNotEqual(sum(g.bids),13)
                self.assertEqual(sum(map(len,g.hands)),0)

    def test_model_roundtrip(self):
        a=Agent()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json';a.save(path)
            b=Agent.load(path)
            np.testing.assert_equal(a.weights,b.weights)

    def test_full_match(self):
        agents=[Agent() for _ in range(4)]
        scores,rounds=run_match(agents,50)
        self.assertEqual(len(rounds),18)
        self.assertEqual(scores,[sum(r['score'][p] for r in rounds) for p in range(4)])

if __name__=='__main__': unittest.main()
