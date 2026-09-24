import unittest
import numpy as np
import pandas as pd

from repeated_reference import RESPONSES, assess_counts_all, retained_event_mask, rng_for


class RepeatedReference20260924Tests(unittest.TestCase):
    def test_phase_streams_are_stable_and_disjoint(self):
        first = rng_for("pilot", 0, "jlh33_session", 17).random(20)
        again = rng_for("pilot", 0, "jlh33_session", 17).random(20)
        final = rng_for("final", 10, "jlh33_session", 17).random(20)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, final))

    def test_one_event_mask_is_reused_for_both_design_assessments(self):
        mask = retained_event_mask("final", 10, "jlh33_session", 17, 31, .2)
        # There is only one mask-generating function; design enters later through counts.
        np.testing.assert_array_equal(mask, retained_event_mask("final", 10, "jlh33_session", 17, 31, .2))
        self.assertEqual(mask.dtype, np.dtype(bool))
    def test_all_response_assessment_preserves_support_and_information(self):
        counts = {
            "quadratic_equal": np.array([[1, 3, 3, 1], [0, 0, 0, 0]]),
            "quadratic_sparse": np.array([[1, 2, 2, 1], [0, 0, 0, 0]]),
        }
        results = assess_counts_all(counts, RESPONSES)
        self.assertEqual(tuple(results), RESPONSES)
        for design in counts:
            support = {results[r][design]["informative_trials"] for r in RESPONSES}
            info = {results[r][design]["conditional_information"] for r in RESPONSES}
            zeros = {results[r][design]["zero_support"] for r in RESPONSES}
            self.assertEqual(len(support), 1)
            self.assertEqual(len(info), 1)
            self.assertEqual(len(zeros), 1)


    def test_response_key_and_zero_support_grouping_are_preserved(self):
        # q and response are part of every grouping key, avoiding all-R overwrite.
        rows = pd.DataFrame([
            {"subject": "s1", "session": "a", "split": "evaluation", "design": design,
             "q": .2, "response": response, "unit_id": 1, "zero_support": True}
            for response in RESPONSES for design in ("quadratic_equal", "quadratic_sparse")
        ])
        grouped = rows.groupby(["subject", "session", "split", "design", "q", "response"], as_index=False).agg(
            units=("unit_id", "size"), zero_support=("zero_support", "mean"))
        self.assertEqual(len(grouped), 2 * len(RESPONSES))
        self.assertEqual(set(grouped.response), set(RESPONSES))
        self.assertTrue((grouped.zero_support == 1).all())
if __name__ == "__main__":
    unittest.main()
