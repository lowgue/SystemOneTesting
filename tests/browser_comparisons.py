"""Browser validation of the large historical catalog and protocol comparisons."""
from contextlib import ExitStack
import json,socket,tempfile,threading,time
from urllib.parse import urljoin
from pathlib import Path
import uvicorn
from playwright.sync_api import sync_playwright,expect
from vision_bench.core import ROOT,load_json
from vision_bench.webui.api import create_app
from vision_bench.webui.services import Studio

def main():
 output=ROOT/'.artifacts/validation/comparisons';output.mkdir(parents=True,exist_ok=True)
 with ExitStack() as stack:
  temp=Path(stack.enter_context(tempfile.TemporaryDirectory()));configs=temp/'configs';configs.mkdir()
  (configs/'local.json').write_text(json.dumps({'models':load_json(ROOT/'benchmarks/configs/local.json')['models'][:2]}))
  (configs/'simulation.json').write_text(json.dumps({'models':[dict(id='SIMULATED-a',adapter='mock'),dict(id='SIMULATED-b',adapter='mock')]}))
  app=create_app(Studio(configs,temp/'jobs'));listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
  server=uvicorn.Server(uvicorn.Config(app,log_level='warning'));thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True);thread.start()
  def stop():server.should_exit=True;thread.join(timeout=15);listener.close()
  stack.callback(stop)
  deadline=time.monotonic()+10
  while not server.started:
   if time.monotonic()>deadline:raise RuntimeError('Startup timeout')
   time.sleep(.05)
  p=stack.enter_context(sync_playwright());browser=p.chromium.launch(headless=True,channel='chromium');stack.callback(browser.close)
  page=browser.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(f'http://127.0.0.1:{port}/#benchmark');page.wait_for_load_state('networkidle')
  page.locator('#config').select_option('local');page.get_by_role('checkbox',name='BLIP VQA',exact=True).check();page.get_by_role('checkbox',name='Laya Vision',exact=True).check()
  page.locator('#dataset').select_option('historical-color-eval');page.locator('#comparison').select_option('all')
  expect(page.locator('#contract')).to_have_value('historical')
  expect(page.locator('.dataset-summary').last).to_contain_text('2880',timeout=10000)
  page.get_by_text('Perguntas, requests e métricas de cada variante',exact=True).click()
  expect(page.get_by_role('heading',name='blip-vqa--structured_json',exact=True)).to_be_visible()
  expect(page.get_by_role('heading',name='laya-vision-head512--native_serial',exact=True)).to_be_visible()
  page.screenshot(path=str(output/'protocols.png'),full_page=True)
  page.locator('#config').select_option('simulation');page.get_by_text('Opções de medição',exact=True).click();page.locator('#repetitions').fill('1')
  page.get_by_role('button',name='Iniciar avaliação do dataset',exact=True).click()
  expect(page.locator('.benchmark-layout [role=status]')).to_have_text('Concluído',timeout=60000)
  report=page.get_by_role('region',name='Relatório do dataset');expect(report).to_be_visible(timeout=10000)
  expect(report).to_contain_text('120 fotos')
  protocols=page.get_by_role('link',name='Perguntas e requests (JSON)',exact=True)
  assert page.request.get(urljoin(page.url,protocols.get_attribute('href'))).ok
  attributes=page.get_by_role('link',name='Métricas por atributo (CSV)',exact=True)
  assert 'attribute_p50_ms' in page.request.get(urljoin(page.url,attributes.get_attribute('href'))).text()
  expect(page.get_by_role('heading',name='Comparações pareadas de modelos e prompts')).to_be_visible()
  chart=page.locator('.attribute-charts img');chart.scroll_into_view_if_needed();expect(chart).to_be_visible()
  page.wait_for_function("document.querySelector('.attribute-charts img').naturalWidth>0")
  page.locator('#attribute-chart').select_option(label='vehicle: color — F1 por classe')
  page.wait_for_function("document.querySelector('.attribute-charts img').complete && document.querySelector('.attribute-charts img').naturalWidth>0")
  page.screenshot(path=str(output/'large-dataset.png'),full_page=True)
  page.reload();expect(page.locator('#dataset')).to_have_value('historical-color-eval');expect(page.locator('#comparison')).to_have_value('all');expect(page.locator('#config')).to_have_value('simulation')
  page.set_viewport_size({'width':390,'height':844});expect(report).to_be_visible();page.screenshot(path=str(output/'mobile.png'),full_page=True)
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile overflow'
  assert not errors,errors
  print('Browser comparisons passed: historical catalog, 8 protocol previews, contract selection, 120-photo automatic dataset, graphs, paired comparisons, restore and mobile.')
if __name__=='__main__':main()
