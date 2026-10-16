"""Exercise the actual guard methods with an interrupted two-bucket apply."""
import ast
import asyncio
import json
from pathlib import Path
s=Path(__file__).resolve().parent
c=s/'full_candidate_v4_atomic'
def method(path,name):
 tree=ast.parse(path.read_text())
 return next(node for cls in tree.body if isinstance(cls,ast.ClassDef) for node in cls.body if isinstance(node,ast.AsyncFunctionDef) and node.name==name)
class SyncBase:
 async def sync_model_from_actor(self):
  self.weights=(1,0)
  self.half.set()
  await self.finish.wait()
  if self.fail:raise RuntimeError('simulated transport failure')
  self.weights=(1,1)
  return 1
nodes=[method(c/'async_grpo/rollout.py','sync_model_from_actor'),method(c/'rlinf/workers/rollout/hf/huggingface_worker.py','_predict_rollout_actions_with_sync_guard')]
cls=ast.ClassDef(name='Fixture',bases=[ast.Name(id='SyncBase',ctx=ast.Load())],keywords=[],body=nodes,decorator_list=[])
tree=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
namespace={'SyncBase':SyncBase}
exec(compile(tree,'actual_candidate_guard_methods','exec'),namespace)
Fixture=namespace['Fixture']
Fixture._predict_rollout_actions=lambda self: self.weights
async def check():
 rows=[]
 for guarded in (False,True):
  w=Fixture();w._full_weight_sync_lock=asyncio.Lock() if guarded else None
  w.weights=(0,0);w.half=asyncio.Event();w.finish=asyncio.Event();w.fail=False
  task=asyncio.create_task(w.sync_model_from_actor())
  await w.half.wait()
  prediction=asyncio.create_task(w._predict_rollout_actions_with_sync_guard())
  await asyncio.sleep(0);await asyncio.sleep(0)
  if guarded:assert not prediction.done(),'prediction observed a partly applied model'
  else:assert prediction.done() and prediction.result()==(1,0)
  w.finish.set();assert await task==1
  result=await prediction
  if guarded:assert result==(1,1) and not w._full_weight_sync_lock.locked()
  rows.append({'guarded':guarded,'prediction':list(result)})
 w=Fixture();w._full_weight_sync_lock=asyncio.Lock();w.weights=(0,0);w.half=asyncio.Event();w.finish=asyncio.Event();w.fail=True;w.finish.set()
 try:await w.sync_model_from_actor()
 except RuntimeError as error:assert str(error)=='simulated transport failure'
 else:raise AssertionError('transport failure was swallowed')
 assert not w._full_weight_sync_lock.locked()
 return {'passed':True,'controls':rows,'transport_failure_propagated_and_lock_released':True,'scope':'Actual candidate entry methods, real asyncio lock, simulated two-bucket apply. Does not verify real distributed communication.'}
result=asyncio.run(check())
(s/'atomic_sync_guard_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
