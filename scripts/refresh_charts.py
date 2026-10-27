"""Refresh presentation assets of completed studio reports; never rerun inference."""
from pathlib import Path
import tempfile,shutil
from vision_bench.report import generate as performance_report
from vision_bench.core import ROOT,load_json,write_json
from vision_bench.dataset_metrics import _quality_chart,_attribute_charts
from vision_bench.simple_charts import generate

def refresh(root):
 for path in Path(root).glob('*/report/evaluation.json'):
  folder=path.parent
  if load_json(folder.parent/'status.json').get('state')!='complete':continue
  data=load_json(path)
  if not all('attribute_latency' in f and 'macro_precision' in f for m in data['models'] for f in m['fields']):
   # Older reports still receive the simple presentation, preserving their metrics.
   data['charts']=[c for c in data.get('charts',[]) if c.get('level')!='simple']
  else:
   _quality_chart(data,folder);_attribute_charts(data,folder)
  data['charts']=generate(data,folder)+data.get('charts',[])
  write_json(path,data)
  with tempfile.TemporaryDirectory() as temporary:
   generated=Path(temporary)/'report'
   performance_report([folder.parent/'run'],generated)
   for chart in generated.glob('comparison_*.*'):shutil.copy2(chart,folder/chart.name)
  print('Charts refreshed:',folder.parent.name,flush=True)

if __name__=='__main__':refresh(ROOT/'.artifacts/studio')
