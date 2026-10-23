import unittest
import numpy as np
from estimation.agent import Agent
from estimation.teach_neural import teacher_target, paired_validation

class NeuralTeachingTests(unittest.TestCase):
    def setUp(self):
        self.base = Agent(np.zeros(31))
        self.x = np.zeros((2,22))

    def test_missing_uncertainty_does_not_teach_overconfident_correction(self):
        target, changed = teacher_target(self.base,self.x,{'teacher_values':[0,100]})
        self.assertFalse(changed)
        np.testing.assert_allclose(target,[.5,.5])

    def test_wrong_pairing_baseline_does_not_teach(self):
        _, changed = teacher_target(self.base,self.x,{'teacher_values':[0,100],
            'paired_standard_error':[0,0], 'baseline_action_index':1})
        self.assertFalse(changed)

    def test_evidence_can_change_target_but_noisy_evidence_cannot(self):
        row = {'teacher_values':[0,10], 'paired_standard_error':[0,1], 'baseline_action_index':0}
        target, changed = teacher_target(self.base,self.x,row)
        self.assertTrue(changed)
        self.assertGreater(target[1],target[0])
        row['paired_standard_error'] = [0,10]
        self.assertFalse(teacher_target(self.base,self.x,row)[1])

    def test_identical_policy_paired_validation_has_zero_delta(self):
        base = Agent()
        gate = paired_validation(base,base,[745000,745001])
        self.assertEqual(gate['paired_improvement_approx_95pct_ci'],[0.,0.])
        self.assertEqual(gate['validation_deals_per_model'],8)

if __name__ == '__main__':
    unittest.main()
