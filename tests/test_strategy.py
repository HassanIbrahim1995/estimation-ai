"""Exact two-trick lessons: continuation matters more than taking this trick.

All unseen cards share the same suit. Thus their allocation among opponents
cannot change these conclusions; they do not rely on a lucky Monte Carlo draw.
"""
import unittest

from estimation.game import Game
from estimation.teacher import Teacher, reconstruct, sample_hands
import random


def two_trick_lesson(required=1):
    g = Game(0, round_index=13)
    g.phase, g.seat, g.trump = 'play', 0, 4
    # Fourth to play: K clubs already wins. Actor holds Q clubs and A clubs.
    # Every opponent has one smaller club left after this trick.
    g.hands = [[10, 12], [0], [1], [2]]
    g.table = [(1, 11), (2, 3), (3, 4)]
    # Eleven legal historical tricks. Nine monochrome D/H/S tricks,
    # then a club trick, then DA wins a mixed trick. No seat 0 wins.
    groups = [list(range(s*13+i, s*13+i+4))
              for s in (1, 2, 3) for i in (0, 4, 8)]
    groups.append([5, 6, 7, 8])
    winners = [1, 1, 2, 2, 2, 2, 3, 3, 3, 1]
    g.played = []
    leader = 1
    for cards, winning_seat in zip(groups, winners):
        assignment = {winning_seat: cards[-1]}
        others = [p for p in range(4) if p != winning_seat]
        assignment.update(zip(others, cards[:-1]))
        g.played.extend(((leader+i)%4, assignment[(leader+i)%4]) for i in range(4))
        leader = winning_seat
    g.played += [(1, 25), (2, 38), (3, 51), (0, 9)] + g.table
    g.voids = [{1}, set(), {1}, {1}]
    g.taken = [0, 4, 4, 3]
    g.bids = [required, 4, 4, 3]
    if required == 2:
        g.bids[3] = 4  # total bids must not equal 13
    g.caller, g.risk_seat = 1, 3
    g.dash = [False]*4
    return g.observe()


class StrategyLessons(unittest.TestCase):
    def test_duck_now_when_one_trick_needed(self):
        # Q loses now; A must win the last trick. Playing A now also makes
        # Q a winner on the final trick and overshoots the one-trick contract.
        o = two_trick_lesson(1)
        for seed in range(4):
            self.assertEqual(Teacher(seed=seed, samples=4).act(o), 10)

    def test_take_now_when_both_tricks_needed(self):
        o = two_trick_lesson(2)
        for seed in range(4):
            self.assertEqual(Teacher(seed=seed, samples=4).act(o), 12)

    def test_lesson_outcomes_hold_for_every_sampled_allocation(self):
        o = two_trick_lesson(1)
        for seed in range(12):
            hands = sample_hands(o, random.Random(seed))
            for action, expected in [(10, 1), (12, 2)]:
                g = reconstruct(o, hands)
                g.step(action)
                while g.phase != 'done':
                    self.assertEqual(len(g.legal()), 1)
                    g.step(g.legal()[0])
                self.assertEqual(g.taken[0], expected)


if __name__ == '__main__':
    unittest.main()
