"""Compare task scope, prompt manipulation and strict output parsing without weights."""
import copy
from pathlib import Path
import unittest
from vision_bench.core import ROOT,load_json,validate_manifest
from vision_bench.protocols import expand_specs,item_contract,describe,output_diagnostics
from vision_bench.webui.services import Studio
from vision_bench.dataset_metrics import paired_comparisons

class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.contract=load_json(ROOT/'benchmarks/contracts/historical_tasks_v1.json')
  self.specs=load_json(ROOT/'benchmarks/configs/local.json')['models'][:2]
 def test_all_variants_have_identical_class_keys_and_explicit_generation(self):
  specs=expand_specs(self.specs,'all');self.assertEqual(len(specs),8)
  for s in specs:
   d=describe(s,self.contract,'vehicle')
   self.assertEqual(d['calls_per_photo'],4 if s['profile'].endswith('serial') else 1)
   if s['adapter']=='laya':self.assertEqual(set(d['request']['questions']['color']['criteria']),set(self.contract['domains']['vehicle']['questions']['color']['criteria']))
   else:self.assertEqual(d['generation']['max_new_tokens'],64 if s['profile']=='structured_json' else 12)
 def test_target_question_and_single_task_are_preserved(self):
  item=dict(domain='vehicle',task_field='color',question='What color is inside the red box?')
  c=item_contract(self.contract,item)
  self.assertEqual(list(c['domains']['vehicle']['questions']),['color'])
  self.assertEqual(c['domains']['vehicle']['prompts']['color'],item['question'])
  self.assertEqual(len(self.contract['domains']['vehicle']['questions']),4)
 def test_json_and_ambiguity_are_not_repaired_silently(self):
  spec=dict(adapter='blip',profile='structured_json')
  for raw in ['red','{"color":"red","extra":1}','{"color":"not_a_class"}']:
   self.assertFalse(output_diagnostics(spec,self.contract,'vehicle',dict(color=raw))['color']['format_valid'])
  self.assertTrue(output_diagnostics(spec,self.contract,'vehicle',dict(color='{"color":"red"}'))['color']['format_valid'])
  self.assertTrue(output_diagnostics(dict(adapter='blip'),self.contract,'vehicle',dict(color='red and blue'))['color']['ambiguous'])
 def test_historical_catalog_is_verified_and_evaluation_split_is_separate(self):
  studio=Studio();catalog={d['id']:d for d in studio.catalog()['datasets']}
  self.assertEqual(catalog['historical-color-all']['events'],340)
  self.assertEqual(catalog['historical-color-eval']['events'],120)
  self.assertEqual(catalog['historical-color-dev']['events'],220)
  for key,path in studio.manifests.items():
   if key.startswith('historical-') and catalog[key]['available']:validate_manifest(path,ROOT,self.contract)
 def test_invalid_comparison_rejected(self):
  with self.assertRaises(ValueError):expand_specs(self.specs,'invented')
 def test_paired_bootstrap_does_not_count_derivatives_as_new_groups(self):
  lookup={str(i):dict(sha256=str(i),group_id='same' if i<2 else 'other') for i in range(3)}
  models=[dict(model=name,photo_results=[dict(id=str(i),domain='vehicle',correct={'color':value}) for i in range(3)]) for name,value in [('a',False),('b',True)]]
  score=paired_comparisons(dict(models=models),lookup)[0]
  self.assertEqual((score['n'],score['groups']),(3,2));self.assertEqual(score['ci95'],[1,1])

if __name__=='__main__':unittest.main()
