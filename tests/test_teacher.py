import random
import unittest
from estimation.agent import Agent
from estimation.game import Game
from estimation.teacher import Teacher,sample_hands,reconstruct

def position(seed=128):
    g=Game(seed,round_index=13)
    bot=Agent()
    while g.phase!='done':
        o=g.observe()
        if o.phase=='play' and sum(o.taken)>=7 and len(o.legal)>1: return g,o
        g.step(bot.act(o))
    raise AssertionError('Expected a nontrivial position')

class TeacherTests(unittest.TestCase):
    def test_samples_preserve_private_hand_voids_and_counts(self):
        g,o=position()
        for seed in range(30):
            hands=sample_hands(o,random.Random(seed))
            self.assertEqual(hands[o.seat],list(o.hand))
            self.assertEqual([len(h) for h in hands],[len(h) for h in g.hands])
            self.assertEqual(sorted(sum(hands,[])+[c for _,c in o.played]),list(range(52)))
            for s in range(4):
                self.assertFalse(any(c//13 in o.voids[s] for c in hands[s]))

    def test_reconstruction_matches_public_observation(self):
        _,o=position()
        copy=reconstruct(o,sample_hands(o,random.Random(7)))
        self.assertEqual(copy.observe(),o)

    def test_teacher_never_reads_real_opponent_hands(self):
        g,o=position()
        a=Teacher(seed=1,samples=3).act(o)
        others=[p for p in range(4) if p!=o.seat]
        g.hands[others[0]],g.hands[others[1]]=g.hands[others[1]],g.hands[others[0]]
        b=Teacher(seed=1,samples=3).act(g.observe())
        self.assertEqual(a,b)
        self.assertIn(a,o.legal)

    def test_sampled_rollout_is_legal_to_terminal(self):
        _,o=position()
        for seed in range(10):
            g=reconstruct(o,sample_hands(o,random.Random(seed)))
            bot=Agent()
            while g.phase!='done': g.step(bot.act(g.observe()))
            self.assertEqual(sum(g.taken),13)
            self.assertEqual(len(set(c for _,c in g.played)),52)

if __name__=='__main__': unittest.main()
