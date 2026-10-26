"""Check the exact rejection boundary against direct small-binomial enumeration."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulate_paired_test_power import critical_values


class ExactPairedTest(unittest.TestCase):
    def test_rejection_matches_enumeration(self):
        for alpha in [.05, .025]:
            boundary = critical_values(100, alpha)
            self.assertEqual(boundary[0], -1)
            for n in [1, 5, 10, 25, 50, 100]:
                for k in range(n // 2 + 1):
                    pvalue = min(1, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)
                    self.assertEqual(k <= boundary[n], pvalue <= alpha, (alpha, n, k))


if __name__ == '__main__':
    unittest.main()
