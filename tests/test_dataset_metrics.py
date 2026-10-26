"""Quality uses independent first-pass events; repetitions only affect performance."""
import copy
import unittest
from vision_bench.core import ROOT,load_json
from vision_bench.dataset_metrics import evaluate,field_scores


class DatasetMetricsTests(unittest.TestCase):
    def setUp(self):
        contract=load_json(ROOT/'benchmarks/contracts/description_ollama_v1.json')
        a=dict(presence='yes',make='audi',color='black',body='sedan')
        b=dict(presence='yes',make='bmw',color='white',body='suv')
        items=[dict(id=str(i),domain='vehicle',rel_path=f'{i}.png',sha256=str(i)) for i in range(4)]
        items[0]['expected']=a;items[1]['expected']=b;items[2]['annotations']={'color':'red'}
        self.frozen=dict(item_ids=['0','1','2','3'],manifest=dict(items=items),contract=contract,repetitions=2,models=[dict(id='m',adapter='mock')])
        self.rows=[]
        for block in range(2):
            for i in range(4):
                error=block==0 and i==1
                pred=copy.copy(a if i!=1 else b)
                if i==2:pred['color']='blue' if block==0 else 'red'
                self.rows.append(dict(request_key=f'{block}/m/{i}',model='m',id=str(i),block=block,predicted=pred,error='failed' if error else None,valid=not error,elapsed_ms=10*(i+1),cpu_s=.01,peak_rss_bytes=1024))

    def test_repetitions_never_inflate_accuracy_or_hide_failures(self):
        result=evaluate(self.frozen,self.rows)['models'][0]
        self.assertEqual(result['requests'],8)
        self.assertEqual(result['photos'],4)
        self.assertEqual(result['annotated_photos'],3)
        self.assertEqual(result['complete_labeled_photos'],2)
        self.assertEqual(result['exact_accuracy'],.5)
        self.assertEqual(result['field_n'],9)
        self.assertAlmostEqual(result['field_accuracy'],4/9)
        self.assertEqual(result['errors'],1)
        self.assertEqual(result['schema_valid_rate'],7/8)
        self.assertEqual(result['latency']['p50_ms'],30)
        color=next(f for f in result['fields'] if f['field']=='color')
        self.assertEqual(color['confusion']['white']['__error__'],1)
        self.assertAlmostEqual(color['macro_f1'],1/3)

    def test_unlabeled_is_none_not_zero(self):
        for item in self.frozen['manifest']['items']:
            item.pop('expected',None);item.pop('annotations',None)
        result=evaluate(self.frozen,self.rows)['models'][0]
        self.assertIsNone(result['exact_accuracy'])
        self.assertIsNone(result['field_accuracy'])
        self.assertEqual(result['field_n'],0)

    def test_all_failures_count_against_quality_without_fake_latency(self):
        for row in self.rows:row.update(error='offline',valid=False)
        result=evaluate(self.frozen,self.rows)['models'][0]
        self.assertEqual(result['exact_accuracy'],0)
        self.assertEqual(result['field_accuracy'],0)
        self.assertEqual(result['errors'],8)
        self.assertIsNone(result['latency']['p50_ms'])

    def test_precision_recall_and_confusion_include_false_positives(self):
        score=field_scores([('yes','yes'),('no','yes'),('yes','__error__')])
        yes=next(c for c in score['classes'] if c['label']=='yes')
        self.assertEqual(yes['precision'],.5)
        self.assertEqual(yes['recall'],.5)
        self.assertEqual(yes['f1'],.5)
        self.assertEqual(score['correct'],1)

    def test_duplicate_records_are_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.frozen,[*self.rows,self.rows[0]])
