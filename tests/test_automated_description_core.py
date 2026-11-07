"""Semantic parser edge cases and renderer invariants; no model download required."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'benchmarks' / 'tasks'))
from automated_description_core import normalize,render,validate,gate

class ContractTests(unittest.TestCase):
    """Group tests for contract tests."""
    def test_negation_is_not_positive(self):
        """Verify that negation is not positive."""
        for raw in ['no helmet','not wearing a helmet']:
            self.assertEqual(normalize('person','helmet',raw),'no')
        self.assertEqual(normalize('person','carrying','no backpack'),'none')
    def test_hard_hat_and_regular_hat_differ(self):
        """Verify that hard hat and regular hat differ."""
        self.assertEqual(normalize('person','helmet','a hard hat'),'yes')
        self.assertEqual(normalize('person','helmet','a hat'),'no')
    def test_unknown_is_not_parse_failure(self):
        """Verify that unknown is not parse failure."""
        for raw in ['', 'car', 'a flying saucer']:
            self.assertEqual(normalize('vehicle','body',raw),'parse_error')
        self.assertEqual(normalize('vehicle','body','unknown'),'unknown')
        self.assertEqual(normalize('person','clothing','not visible'),'not_visible')
    def test_word_boundaries_and_conflict(self):
        """Verify that word boundaries and conflict."""
        self.assertEqual(normalize('vehicle','make','oxford'),'parse_error')
        self.assertEqual(normalize('vehicle','color','black and white'),'parse_error')
        self.assertEqual(normalize('person','clothing','t-shirt'),'shirt_tshirt')
    def test_schema_rejects_out_of_taxonomy(self):
        """Verify that schema rejects out of taxonomy."""
        self.assertFalse(validate('vehicle',dict(presence='yes',make='other',body='suv',color='blue')))
    def test_absence_filter_does_not_mutate_raw(self):
        """Verify that absence filter does not mutate raw."""
        raw=dict(presence='no',make='ford',body='truck',color='red')
        filtered=gate('vehicle',raw)
        self.assertEqual(raw['make'],'ford')
        self.assertTrue(validate('vehicle',filtered))
        self.assertIsNone(render('vehicle',filtered)['text'])
    def test_render_does_not_invent_unknowns(self):
        """Verify that render does not invent unknowns."""
        ans=dict(presence='yes',make='unknown',body='suv',color='parse_error')
        out=render('vehicle',ans)
        self.assertEqual(out['facts'],{'body':'suv'})
        self.assertNotIn('unknown',out['text'])
        self.assertNotIn('parse_error',out['text'])

if __name__=='__main__':unittest.main()
