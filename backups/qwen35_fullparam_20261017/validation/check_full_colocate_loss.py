"""Exercise the real sync microbatch path with an extreme negative-advantage case."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from rlinf.scheduler import Worker
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.logp=torch.nn.Parameter(torch.full((1,16),-1.0))
 def forward(self,**kwargs):return {'logprobs':self.logp,'values':None}
rows=[]
for full in (False,True):
 cfg=OmegaConf.create({'actor':{'model':{'model_type':'alpamayo2','full_parameter_training':full,'action_dim':2}},'rollout':{'sampling_params':{'temperature_train':0.6,'top_k':0}},'algorithm':{'adv_type':'grpo','loss_type':'actor','logprob_type':'chunk_level','reward_type':'chunk_level','entropy_bonus':0,'clip_ratio_high':0.1,'clip_ratio_low':0.1,'clip_ratio_c':3.0,'value_clip':0.2,'huber_delta':10.0},'env':{'train':{'max_episode_steps':128}},'runner':{'task_type':'embodied'}})
 actor=SimpleNamespace(cfg=cfg,device=torch.device('cpu'),model=Model(),before_micro_batch=lambda *a,**k:nullcontext(),amp_context=nullcontext(),optimizer_steps=0,critic_warmup_steps=0,gradient_accumulation=1,enable_sft_co_train=False,grad_scaler=SimpleNamespace(scale=lambda x:x))
 batch={'advantages':torch.full((1,1),-1.0),'prev_logprobs':torch.full((1,16),-3.0),'forward_inputs':{}}
 metrics={}
 with patch.object(Worker,'torch_platform',SimpleNamespace(current_device=lambda:'cpu')):
  EmbodiedFSDPActor.train_micro_batch(actor,batch,metrics,is_last=True)
 grad=float(actor.model.logp.grad.abs().max())
 rows.append({'full_parameter_training':full,'maximum_gradient':grad,'total_loss':metrics['actor/total_loss'][0]})
assert rows[0]['maximum_gradient']>0
assert rows[1]['maximum_gradient']==0
result={'passed':True,'cases':rows,'scope':'Real sync actor microbatch and registered actor loss on CPU. Full-mode dual clip caps extreme negative-advantage gradients, historical non-full default remains unchanged.'}
s=Path(__file__).resolve().parent
(s/'full_colocate_loss_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
