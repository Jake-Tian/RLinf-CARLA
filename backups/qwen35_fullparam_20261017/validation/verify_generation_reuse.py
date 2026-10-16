"""Check integrated inference reuse before and after a vision weight change."""
import json,os,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from transformers import LogitsProcessor,LogitsProcessorList
from alpamayo_carla.policy import AlpamayoPolicy
s=Path(__file__).resolve().parent
out=s/('generation_reuse_check_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=True)
os.environ['RLINF_FULL_AUDIT']='0'
cfg=OmegaConf.load(s/'configs/carla_grpo_qwen35_full_async_v6.yaml')
model=AlpamayoPolicy(cfg.actor.model,torch.bfloat16).cuda().eval()
assert all(p.requires_grad for p in model.parameters())
visual=model.backbone.vlm.model.visual
original=visual.forward
class Replay(LogitsProcessor):
 def __init__(self,tokens):self.tokens=tokens;self.lp=[]
 def __call__(self,ids,scores):
  action=self.tokens[:,len(self.lp):len(self.lp)+1]
  lp=(scores[:,model.start:model.start+model.bins].float()/model.temperature).log_softmax(-1)
  self.lp.append(lp.gather(1,action-model.start).squeeze(1))
  forced=torch.full_like(scores,-torch.inf);forced.scatter_(1,action,0);return forced
rows=[];reference_before={};changed=False
with torch.no_grad():
 for version in (0,1):
  if version:
   parameter=next(visual.parameters())
   parameter.reshape(-1)[:64].add_(0.01)
  for rank in (0,3):
   saved=torch.load(s/f'bootstrap_L_18867/first_microbatch_rank{rank}.pt',map_location='cpu',weights_only=True)
   data={k:v.cuda() for k,v in saved['forward_inputs'].items()}
   ref=model(forward_inputs=data)['logprobs'].cpu().float()
   if version==0:
    assert torch.equal(ref,saved['training_logprobs'].float())
    reference_before[rank]=ref
   else:changed |= not torch.equal(ref,reference_before[rank])
   for reuse in (False,True):
    model.reuse_vision_within_generation=reuse
    calls=[0]
    def observed(*args,**kwargs):
     calls[0]+=1
     return original(*args,**kwargs)
    visual.forward=observed
    prompt=data['input_ids'].long()
    inputs=model.backbone.prepare_multimodal_model_inputs(prompt,{'attention_mask':data['attention_mask'],'pixel_values':data['pixel_values'].flatten(0,1),'image_grid_thw':data['image_grid_thw'].flatten(0,1)})
    processor=Replay(data['action_tokens'].long())
    torch.cuda.synchronize();started=time.perf_counter()
    try:
     generated=model._generate_action_tokens(input_ids=prompt,**inputs,max_new_tokens=model.token_count,min_new_tokens=model.token_count,do_sample=False,use_cache=False,logits_processor=LogitsProcessorList([processor]))
     torch.cuda.synchronize();elapsed=time.perf_counter()-started
     assert visual.forward is observed,'temporary forward hook was not restored'
    finally:visual.forward=original
    assert torch.equal(generated[:,prompt.shape[1]:],data['action_tokens'])
    lp=model._tick_logprobs(torch.stack(processor.lp,1)).cpu().float()
    delta=float((lp-ref).abs().max())
    assert delta==0,(version,rank,reuse,delta)
    assert calls[0]==(1 if reuse else model.token_count),calls
    row={'weight_version':version,'rank_input':rank,'reuse':reuse,'elapsed_s':elapsed,'vision_executions':calls[0],'logprob_vs_current_native_teacher_forcing_max':delta}
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
assert changed,'weight perturbation had no observed output effect'
assert all(p.requires_grad for p in model.parameters())
(out/'passed.json').write_text(json.dumps({'passed':True,'vision_weight_change_changed_output':changed,'all_parameters_still_trainable':True,'scope':'Integrated generate helper on two real observations, original and changed vision weights. No cross-generation reuse, no training data cache, no online throughput claim.'},indent=2)+'\n')
