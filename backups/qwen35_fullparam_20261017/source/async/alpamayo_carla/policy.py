"""Alpamayo2 token policy implementing RLinf's embodied model interface."""
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutputWithPooling

from autopilot.models.alpamayo2.sft_base_model import TrainableReasoningVLA
from autopilot.utils.dataset.qwen_processor import QwenProcessor
from rlinf.models.embodiment.base_policy import BasePolicy


class ActionTokens(LogitsProcessor):
    """Restrict each executed action token to the checkpoint's original bins."""

    def __init__(self, start, bins):
        self.start, self.bins = start, bins

    def __call__(self, input_ids, scores):
        allowed = scores[:, self.start:self.start+self.bins].clone()
        scores.fill_(-torch.inf)
        scores[:, self.start:self.start+self.bins] = allowed
        return scores


def prepare_observation_batch(processor, preprocess, env_obs, index, device="cpu"):
    """Build the full sample required by the checkpoint's original collator."""
    indices = [index] if isinstance(index, int) else index
    samples = []
    for i in indices:
        sample = {
            "image_frames": torch.as_tensor(env_obs["image_frames"][i], device=device),
            "camera_indices": torch.tensor([0, 1, 2, 6]),
            "relative_timestamps": torch.arange(4).float().mul(0.1).expand(4, -1),
            "ego_history_xyz": torch.as_tensor(env_obs["ego_history_xyz"][i], device="cpu"),
            "ego_history_rot": torch.as_tensor(env_obs["ego_history_rot"][i], device="cpu"),
        }
        sample["tokenized_data"] = preprocess(sample)
        samples.append(sample)
    return processor.collate_fn(samples)


class AlpamayoPolicy(torch.nn.Module, BasePolicy):
    """Generate only the executed action prefix, retaining the trained token ABI."""

    def __init__(self, cfg, dtype):
        super().__init__()
        self.backbone = TrainableReasoningVLA.from_alpamayo_checkpoint(
            cfg.model_path, dtype=dtype, attn_implementation="flash_attention_2",
            print_param_count=False)
        self.config = self.backbone.vlm.config
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.repeat = int(cfg.get("control_repeat", 2))
        if self.num_action_chunks % self.repeat:
            raise ValueError("action chunk must contain whole 0.1-second controls")
        self.native_steps = self.num_action_chunks // self.repeat
        if not 0 < self.native_steps <= 64:
            raise ValueError("Alpamayo action prefix must contain 1 to 64 native steps")
        self.token_count = self.native_steps * 2
        self.start = self.backbone.future_token_start_idx
        self.bins = self.backbone.traj_tokenizer.vocab_size
        self.temperature = float(cfg.get("temperature", 0.6))
        self.generation_use_cache = bool(cfg.get("generation_use_cache", True))
        self.reuse_vision_within_generation = bool(cfg.get("reuse_vision_within_generation", False))
        self.rollout_micro_batch_size = int(cfg.get("rollout_micro_batch_size", 1))
        if self.rollout_micro_batch_size < 1:
            raise ValueError("rollout_micro_batch_size must be positive")
        self.preprocess_on_gpu = bool(cfg.get("preprocess_on_gpu", False))
        self.processor = QwenProcessor(cfg.model_path,
            traj_vocab_size=self.backbone.config.traj_vocab_size,
            min_pixels=self.backbone.config.min_pixels, max_pixels=self.backbone.config.max_pixels,
            include_camera_ids=True, include_frame_nums=True)
        self.preprocess = self.processor.get_preprocess_data_fn(
            num_tokens_per_history_traj=48, num_tokens_per_future_traj=128,
            components_order=["image", "traj_history", "prompt", "traj_future"],
            components_prompt=["traj_future"], label_components=["traj_future"], generation_mode=True)
        if cfg.get("full_parameter_training", False):
            if cfg.get("reuse_frozen_vision", False):
                raise ValueError("full-parameter training cannot reuse frozen vision")
            self.backbone.requires_grad_(True)
        else:
            self.backbone.vlm = get_peft_model(self.backbone.vlm,
                LoraConfig(r=int(cfg.get("alpamayo_lora_rank", 8)), lora_alpha=16,
                           lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none"))
        import os
        initial_path = os.environ.get("RLINF_POLICY_INITIAL_LORA")
        if initial_path:
            import hashlib
            import json
            from pathlib import Path
            initial = torch.load(initial_path, map_location="cpu", weights_only=True)
            parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
            assert parameters.keys() == initial.keys(), "Initial LoRA keys differ"
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(initial[name].to(parameter))
            def fingerprint(items):
                digest = hashlib.sha256()
                count = size = 0
                for name, value in items:
                    value = value.detach().cpu().contiguous()
                    digest.update(name.encode())
                    digest.update(str(value.dtype).encode())
                    digest.update(str(tuple(value.shape)).encode())
                    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                    count += 1
                    size += value.numel() * value.element_size()
                return {"sha256": digest.hexdigest(), "tensors": count, "bytes": size}
            frozen = fingerprint((n, p) for n, p in self.named_parameters() if not p.requires_grad)
            trainable = fingerprint(parameters.items())
            persistent = fingerprint(self.state_dict().items()) if os.environ.get("RLINF_AUDIT_FULL_STATE") == "1" else None
            directory = Path(os.environ["CARLA_OUTPUT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"policy_initial_{os.getpid()}.json").write_text(json.dumps({
                "frozen": frozen, "trainable": trainable, "full": persistent,
                "initial_file_sha256": hashlib.sha256(Path(initial_path).read_bytes()).hexdigest(),
                "trainable_keys": list(parameters)}, indent=2))
        if cfg.get("full_parameter_training", False):
            import hashlib
            import json
            import os
            from pathlib import Path
            if os.environ.get("RLINF_POLICY_INITIAL_LORA"):
                raise ValueError("LoRA initialization is invalid for full-parameter training")
            parameters = list(self.named_parameters())
            assert all(parameter.requires_grad for _, parameter in parameters)
            assert not any("lora_" in name for name, _ in parameters)
            if os.environ.get("RLINF_FULL_AUDIT") == "1":
                digest = hashlib.sha256()
                rows = []
                for name, parameter in parameters:
                    value = parameter.detach().cpu().to(torch.bfloat16).contiguous()
                    digest.update(name.encode())
                    digest.update(str(tuple(value.shape)).encode())
                    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                    rows.append({"name": name, "numel": parameter.numel(),
                                 "dtype": str(parameter.dtype), "requires_grad": parameter.requires_grad})
                directory = Path(os.environ["CARLA_OUTPUT_DIR"])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"policy_full_initial_{os.getpid()}.json").write_text(json.dumps({
                    "canonical_bf16_parameter_sha256": digest.hexdigest(),
                    "total_parameters": sum(row["numel"] for row in rows),
                    "trainable_parameters": sum(row["numel"] for row in rows if row["requires_grad"]),
                    "parameters": rows, "full_parameter_training": True,
                    "initialization": "native SFT checkpoint, no LoRA"}, indent=2))
        self.reuse_frozen_vision = bool(cfg.get("reuse_frozen_vision", False))
        self._cached_vision_features = None
        self._capture_vision_features = False
        self._rollout_image_features = None
        if self.reuse_frozen_vision:
            visual = self.backbone.vlm.get_base_model().model.visual
            if any(p.requires_grad for p in visual.parameters()) or any(
                    isinstance(m, torch.nn.Dropout) and m.p for m in visual.modules()):
                raise ValueError("vision reuse requires frozen deterministic vision")
            original_visual_forward = visual.forward
            def forward_with_features(*args, **kwargs):
                if self._cached_vision_features is not None:
                    return BaseModelOutputWithPooling(pooler_output=self._cached_vision_features)
                output = original_visual_forward(*args, **kwargs)
                if self._capture_vision_features:
                    self._rollout_image_features = output.pooler_output.detach()
                return output
            visual.forward = forward_with_features


    @property
    def device(self):
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.vlm.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        return self.default_forward(**kwargs)

    def _tick_logprobs(self, logprobs):
        native = logprobs.reshape(-1, self.native_steps, 2)
        # Repeated simulator ticks do not correspond to new independent samples.
        result = native.new_zeros((native.shape[0], self.num_action_chunks, 2))
        result[:, ::self.repeat] = native
        return result.flatten(1)

    def _controls(self, tokens):
        cfg = self.backbone.config.traj_tokenizer_cfg
        space = cfg["action_space_cfg"]
        normalized = (tokens.cpu().numpy()-self.start)/(self.bins-1)
        normalized = normalized.reshape(-1, self.native_steps, 2)
        lower, upper = np.asarray(cfg["dims_min"]), np.asarray(cfg["dims_max"])
        normalized = lower + normalized*(upper-lower)
        mean = np.array([space["accel_mean"], space["curvature_mean"]])
        std = np.array([space["accel_std"], space["curvature_std"]])
        controls = normalized*std+mean
        controls[...,0] = np.clip(controls[...,0], *space["accel_bounds"])
        controls[...,1] = np.clip(controls[...,1], *space["curvature_bounds"])
        return np.repeat(controls, self.repeat, axis=1).astype(np.float32)

    def _generate_action_tokens(self, **kwargs):
        if not self.reuse_vision_within_generation:
            return self.backbone.vlm.generate(**kwargs)
        if torch.is_grad_enabled():
            raise RuntimeError("generation vision reuse requires no_grad")
        vlm = self.backbone.vlm.get_base_model() if hasattr(self.backbone.vlm, "get_base_model") else self.backbone.vlm
        visual = vlm.model.visual
        if visual.training:
            raise RuntimeError("generation vision reuse requires evaluation mode")
        original = visual.forward
        features = None

        def within_generation(*args, **kwargs):
            nonlocal features
            if features is not None:
                return BaseModelOutputWithPooling(pooler_output=features)
            output = original(*args, **kwargs)
            features = output.pooler_output
            return output

        visual.forward = within_generation
        try:
            return self.backbone.vlm.generate(**kwargs)
        finally:
            visual.forward = original
            features = None

    @torch.no_grad()
    def predict_action_batch(self, env_obs, calculate_logprobs=True,
                             calculate_values=False, mode="train", **kwargs):
        outputs, controls, previous = [], [], []
        batch_size = env_obs["image_frames"].shape[0]
        for i in range(0, batch_size, self.rollout_micro_batch_size):
            end = min(i + self.rollout_micro_batch_size, batch_size)
            size = end - i
            batch = prepare_observation_batch(self.processor, self.preprocess, env_obs, range(i,end), device=self.device if self.preprocess_on_gpu else "cpu")
            data = {key:value.to(self.device) for key,value in batch["tokenized_data"].items()}
            prompt = self.backbone.fuse_traj_tokens(data.pop("input_ids"), {
                "ego_history_xyz":batch["ego_history_xyz"].to(self.device).unsqueeze(1),
                "ego_history_rot":batch["ego_history_rot"].to(self.device).unsqueeze(1)})
            inputs = self.backbone.prepare_multimodal_model_inputs(prompt, data)
            self._rollout_image_features = None
            self._capture_vision_features = self.reuse_frozen_vision
            try:
                sampled = self._generate_action_tokens(input_ids=prompt, **inputs,
                    max_new_tokens=self.token_count, min_new_tokens=self.token_count,
                    do_sample=kwargs.get("do_sample", mode in {"train","training"}), temperature=self.temperature,
                    top_k=0, top_p=1.0, use_cache=self.generation_use_cache,
                    logits_processor=LogitsProcessorList([ActionTokens(self.start,self.bins)]),
                    return_dict_in_generate=True, output_scores=True)
            finally:
                self._capture_vision_features = False
            tokens = sampled.sequences[:,prompt.shape[1]:]
            if tokens.shape[1] != self.token_count or not bool(
                    ((tokens>=self.start)&(tokens<self.start+self.bins)).all()):
                raise ValueError("invalid Alpamayo action prefix")
            logprobs = torch.stack([torch.log_softmax(scores[:,self.start:self.start+self.bins].float(),-1)
                .gather(1,tokens[:,j:j+1]-self.start).squeeze(1)
                for j,scores in enumerate(sampled.scores)],dim=1)
            previous.append(self._tick_logprobs(logprobs))
            controls.append(self._controls(tokens))
            outputs.append({"input_ids":prompt.detach().cpu(),
                "attention_mask":batch["tokenized_data"]["attention_mask"].detach().cpu(),
                "pixel_values":None if self.reuse_frozen_vision else batch["tokenized_data"]["pixel_values"].reshape(size,-1,batch["tokenized_data"]["pixel_values"].shape[-1]).detach().cpu(),
                "image_grid_thw":batch["tokenized_data"]["image_grid_thw"].reshape(size,-1,3).detach().cpu(),
                "action_tokens":tokens.detach().cpu()})
            if self.reuse_frozen_vision:
                if self._rollout_image_features is None:
                    raise RuntimeError("generation did not produce vision features")
                outputs[-1].pop("pixel_values")
                features = self._rollout_image_features
                outputs[-1]["image_features"] = features.reshape(size, -1, features.shape[-1]).cpu()
                self._rollout_image_features = None
        forward_inputs = {key:torch.cat([row[key] for row in outputs]) for key in outputs[0]}
        return np.concatenate(controls), {"prev_logprobs":torch.cat(previous).cpu(),
            "prev_values":None,"forward_inputs":forward_inputs}

    def default_forward(self, forward_inputs, compute_logprobs=True,
                        compute_entropy=False, compute_values=False, loss_mask=None, **kwargs):
        if compute_values:
            raise NotImplementedError("Alpamayo2 adapter supports actor-only GRPO")
        if loss_mask is not None:
            active = loss_mask.bool().reshape(loss_mask.shape[0], -1).any(-1)
            if not bool(active.all()):
                indices = active.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    # Every FSDP rank still traverses the same wrapped layers.
                    indices = torch.zeros(1, dtype=torch.long, device=active.device)
                selected = {key:value.index_select(0,indices) for key,value in forward_inputs.items()}
                output = self.default_forward(selected, compute_logprobs, compute_entropy, False)
                restored = {}
                for key,value in output.items():
                    if value is None:
                        restored[key] = None
                    else:
                        full = value.new_zeros((active.shape[0], *value.shape[1:]))
                        full = full.index_copy(0,indices,value)
                        restored[key] = full * active.reshape(-1,*([1]*(value.ndim-1)))
                return restored
        tokens = forward_inputs["action_tokens"].long()
        ids = torch.cat([forward_inputs["input_ids"],tokens],dim=1).long()
        attention = torch.cat([forward_inputs["attention_mask"],torch.ones_like(tokens)],dim=1)
        features = forward_inputs.get("image_features")
        if features is not None and not self.reuse_frozen_vision:
            raise ValueError("image_features require reuse_frozen_vision")
        if features is not None:
            from rlinf.hybrid_engines.fsdp import FSDPModule
            visual = self.backbone.vlm.get_base_model().model.visual
            if any(isinstance(block, FSDPModule) for block in visual.blocks):
                raise ValueError("native vision reuse requires VisionBlock outside FSDP input casting")
        inputs = {"attention_mask":attention,
            "pixel_values":features.new_empty((0,)) if features is not None else forward_inputs["pixel_values"].flatten(0,1),
            "image_grid_thw":forward_inputs["image_grid_thw"].flatten(0,1)}
        inputs = self.backbone.prepare_multimodal_model_inputs(ids,inputs)
        self._cached_vision_features = features.flatten(0,1) if features is not None else None
        try:
            output = self.backbone.vlm(input_ids=ids,**inputs,use_cache=False,
                                       logits_to_keep=self.token_count+1)
        finally:
            self._cached_vision_features = None
        logits = output.logits[:,:-1,self.start:self.start+self.bins].float()/self.temperature
        logp = torch.log_softmax(logits,-1)
        chosen = logp.gather(-1,(tokens-self.start).unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp()*logp).sum(-1) if compute_entropy else None
        return {"logprobs":self._tick_logprobs(chosen),
                "entropy":self._tick_logprobs(entropy) if entropy is not None else None,
                "values":None}


def get_model(cfg, torch_dtype):
    """Construct the policy through RLinf's existing model registry."""
    return AlpamayoPolicy(cfg, torch_dtype)
