"""
LoRA Fine-tuning for the PassiveQA Decision Planner.

Model : mistralai/Mistral-7B-Instruct-v0.3
Task  : ANSWER / ASK / ABSTAIN routing decision
GPU   : L4 (24 GB VRAM) | bf16 | no quantisation
LoRA  : r=32, alpha=64, target all projection layers
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections import defaultdict
from typing import Optional

import torch


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_R      = 32
LORA_ALPHA  = 64
LORA_DROP   = 0.05
BATCH_SIZE  = 8
GRAD_ACCUM  = 2        # effective batch = 16
MAX_SEQ_LEN = 512
EPOCHS      = 2
LR          = 2e-4


# ── Dialogue-preserving subsampler ────────────────────────────────────────────

def subsample_preserving_dialogues(
    path: str,
    target_total: int,
    action_ratios: Optional[dict] = None,
) -> str:
    """
    Sample `target_total` samples from a JSONL file while keeping
    multi-turn dialogues intact (all turns of a chosen dialogue are
    included or excluded together).

    Returns path to the subsampled JSONL file (suffix _sub.jsonl).
    """
    import random
    random.seed(42)

    if action_ratios is None:
        action_ratios = {"ANSWER": 0.30, "ASK": 0.38, "ABSTAIN": 0.32}

    by_dialogue:  dict = defaultdict(list)
    single_turn:  list = []

    with open(path) as f:
        for line in f:
            s   = json.loads(line)
            dlg = s.get("dialogue_id")
            (by_dialogue[dlg] if dlg else single_turn).append(s)

    def _get_action(s: dict) -> str:
        c = s["messages"][2]["content"]
        if "<decision>\nANSWER" in c: return "ANSWER"
        if "<decision>\nASK"    in c: return "ASK"
        return "ABSTAIN"

    dlg_by_action:    dict = defaultdict(list)
    single_by_action: dict = defaultdict(list)

    for dlg_id, turns in by_dialogue.items():
        acts = [_get_action(t) for t in turns]
        dlg_by_action[max(set(acts), key=acts.count)].append(dlg_id)

    for s in single_turn:
        single_by_action[_get_action(s)].append(s)

    selected = []
    for action, ratio in action_ratios.items():
        target = int(target_total * ratio)
        count  = 0

        dlg_ids = dlg_by_action.get(action, [])
        random.shuffle(dlg_ids)
        for dlg_id in dlg_ids:
            turns = by_dialogue[dlg_id]
            if count + len(turns) <= target:
                selected.extend(turns)
                count += len(turns)
            if count >= target:
                break

        singles = single_by_action.get(action, [])
        random.shuffle(singles)
        rem = target - count
        if rem > 0:
            selected.extend(singles[:rem])
        print(f"  {action:10}: {min(count + rem, target)} samples")

    random.shuffle(selected)
    tmp = path.replace(".jsonl", "_sub.jsonl")
    with open(tmp, "w") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")
    print(f"  Total: {len(selected)} → {tmp}")
    return tmp


# ── Callback ──────────────────────────────────────────────────────────────────

class PlannerCallback:
    """
    TrainerCallback that logs progress and saves adapter to Drive after
    every epoch so nothing is lost mid-training.
    """

    def __init__(self, model, tokenizer, gdrive_path: str):
        self.model       = model
        self.tokenizer   = tokenizer
        self.gdrive_path = gdrive_path
        self.train_losses: list  = []
        self.eval_losses:  list  = []
        self.best_eval    = float("inf")
        self.best_epoch   = 0

    def as_hf_callback(self):
        """Return a HuggingFace TrainerCallback wrapping this object."""
        from transformers import TrainerCallback

        parent = self

        class _Callback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                step      = state.global_step
                loss      = logs.get("loss")
                eval_loss = logs.get("eval_loss")
                lr        = logs.get("learning_rate", 0)
                if loss is not None:
                    parent.train_losses.append((step, loss))
                    if step % 100 == 0 or step <= 20:
                        print(f"  step {step:>5} | "
                              f"train_loss: {loss:.4f} | lr: {lr:.2e}")
                if eval_loss is not None:
                    parent.eval_losses.append((step, eval_loss))
                    marker = ""
                    if eval_loss < parent.best_eval:
                        parent.best_eval  = eval_loss
                        parent.best_epoch = state.epoch
                        marker            = "  ← best ✅"
                    print(f"\n  EPOCH {state.epoch:.0f} | "
                          f"eval_loss: {eval_loss:.4f}{marker}\n")

            def on_epoch_begin(self, args, state, control, **kwargs):
                print(f"\n  ════ EPOCH {int(state.epoch)+1} / "
                      f"{int(args.num_train_epochs)} ════\n")

            def on_epoch_end(self, args, state, control, **kwargs):
                epoch_dir = os.path.join(
                    parent.gdrive_path, f"epoch_{int(state.epoch)}"
                )
                os.makedirs(epoch_dir, exist_ok=True)
                try:
                    parent.model.save_pretrained(epoch_dir)
                    parent.tokenizer.save_pretrained(epoch_dir)
                    print(f"\n  💾  Epoch {int(state.epoch)} adapter → {epoch_dir}")
                except Exception as e:
                    print(f"\n  ⚠️  Drive save failed: {e}")

            def on_train_end(self, args, state, control, **kwargs):
                print("\n" + "=" * 60)
                print("  TRAINING COMPLETE")
                print("=" * 60)
                if parent.train_losses:
                    print(f"  Final train loss : {parent.train_losses[-1][1]:.4f}")
                print(f"  Best eval loss   : {parent.best_eval:.4f} "
                      f"(epoch {parent.best_epoch:.0f})")
                print(f"  Total steps      : {state.global_step}")

        return _Callback()


# ── Main training function ────────────────────────────────────────────────────

def finetune_planner(
    train_file:   str,
    val_file:     str,
    output_dir:   str,
    adapter_dir:  str,
    gdrive_path:  str,
    model_id:     str = MODEL_ID,
    epochs:       int = EPOCHS,
    batch_size:   int = BATCH_SIZE,
    grad_accum:   int = GRAD_ACCUM,
    lr:           float = LR,
    max_seq_len:  int = MAX_SEQ_LEN,
    lora_r:       int = LORA_R,
    lora_alpha:   int = LORA_ALPHA,
    subsample_train: Optional[int] = 9000,
    subsample_val:   Optional[int] = 1200,
) -> dict:
    """
    Fine-tune Mistral-7B-Instruct with LoRA on the PassiveQA planner task.

    Args:
        train_file:      Path to ft_train_mistral.jsonl.
        val_file:        Path to ft_val_mistral.jsonl.
        output_dir:      Local checkpoint directory.
        adapter_dir:     Local final adapter directory.
        gdrive_path:     Google Drive path for epoch saves.
        model_id:        HuggingFace model identifier.
        epochs:          Number of training epochs.
        batch_size:      Per-device batch size.
        grad_accum:      Gradient accumulation steps.
        lr:              Peak learning rate.
        max_seq_len:     Maximum token sequence length.
        lora_r:          LoRA rank.
        lora_alpha:      LoRA scaling factor.
        subsample_train: Dialogue-preserving subsample size for train (None = all).
        subsample_val:   Dialogue-preserving subsample size for val (None = all).

    Returns:
        Dict with training metrics and file paths.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    for d in [output_dir, adapter_dir]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(gdrive_path, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Switch to a GPU runtime.")

    gpu_name   = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_bf16   = torch.cuda.is_bf16_supported()
    use_fp16   = not use_bf16

    print(f"\n  GPU  : {gpu_name}  ({gpu_mem_gb:.1f} GB)")
    print(f"  bf16 : {use_bf16}")

    # ── Subsample ────────────────────────────────────────────────
    if subsample_train:
        print(f"\nSubsampling train → {subsample_train} …")
        train_file = subsample_preserving_dialogues(
            train_file, target_total=subsample_train
        )
    if subsample_val:
        print(f"Subsampling val → {subsample_val} …")
        val_file = subsample_preserving_dialogues(
            val_file, target_total=subsample_val
        )

    # ── Load tokenizer + model ────────────────────────────────────
    print(f"\n  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype       = torch.bfloat16 if use_bf16 else torch.float16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # ── Datasets ──────────────────────────────────────────────────
    train_ds_raw = load_dataset("json", data_files=train_file, split="train")
    val_ds_raw   = load_dataset("json", data_files=val_file,   split="train")
    print(f"\n  Train: {len(train_ds_raw)}  Val: {len(val_ds_raw)}")

    def _format(example):
        text   = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(
            text, truncation=True, max_length=max_seq_len, return_tensors=None
        )
        return {"text": tokenizer.decode(
            tokens["input_ids"], skip_special_tokens=False
        )}

    train_ds = train_ds_raw.map(_format, batched=False)
    val_ds   = val_ds_raw.map(_format, batched=False)

    # ── LoRA ──────────────────────────────────────────────────────
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r              = lora_r,
        lora_alpha     = lora_alpha,
        lora_dropout   = LORA_DROP,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
        target_modules = [
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\n  LoRA trainable: {trainable:,}  ({100*trainable/all_params:.3f}%)")

    # ── Trainer ───────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        logging_steps               = 20,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        dataloader_pin_memory       = False,
        optim                       = "adamw_torch",
        max_grad_norm               = 1.0,
        weight_decay                = 0.01,
        dataset_text_field          = "text",
    )

    cb = PlannerCallback(model, tokenizer, gdrive_path)

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        args             = training_args,
        callbacks        = [cb.as_hf_callback()],
    )

    gc.collect()
    torch.cuda.empty_cache()

    # ── Train ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING TRAINING")
    print("=" * 60 + "\n")
    train_result = trainer.train()

    # ── Save adapter ──────────────────────────────────────────────
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    final_drive_dir = os.path.join(gdrive_path, "final_adapter")
    os.makedirs(final_drive_dir, exist_ok=True)
    shutil.copytree(adapter_dir, final_drive_dir, dirs_exist_ok=True)

    m = train_result.metrics
    config_record = {
        "model_id"            : model_id,
        "lora_r"              : lora_r,
        "lora_alpha"          : lora_alpha,
        "epochs"              : epochs,
        "batch_size"          : batch_size,
        "grad_accum"          : grad_accum,
        "lr"                  : lr,
        "max_seq_len"         : max_seq_len,
        "train_samples"       : len(train_ds),
        "val_samples"         : len(val_ds),
        "best_eval_loss"      : cb.best_eval,
        "best_epoch"          : cb.best_epoch,
        "final_train_loss"    : m.get("train_loss", 0),
        "train_runtime_min"   : m.get("train_runtime", 0) / 60,
    }
    cfg_path = os.path.join(final_drive_dir, "training_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"\n  ✅ Adapter → {adapter_dir}/")
    print(f"  ✅ Drive   → {final_drive_dir}/")
    print(f"  ✅ Config  → {cfg_path}")

    return config_record


def push_to_hub(
    adapter_dir: str,
    hf_token:    str,
    repo_id:     str = "Moodlerz/mistral-planner-aaqa",
) -> None:
    """Push the LoRA adapter to the Hugging Face Hub."""
    from huggingface_hub import HfApi, login

    login(token=hf_token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=hf_token,
                    exist_ok=True, private=False)

    model_card = f"""---
base_model: {MODEL_ID}
tags:
- peft
- lora
- question-answering
- decision-planner
- knowledge-graph
license: apache-2.0
---

# Mistral-7B — KG-Grounded Decision Planner (PassiveQA)

LoRA adapter fine-tuned on Mistral-7B-Instruct-v0.3 for
knowledge-graph-grounded epistemic decision planning.

## Task
Given a user query and KG triples, the model decides:
- **ANSWER** — graph has a complete reasoning path
- **ASK**    — graph is partial; targeted clarification resolves the gap
- **ABSTAIN**— topic is absent from the graph entirely

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer  = AutoTokenizer.from_pretrained("{repo_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    "{MODEL_ID}", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
model.eval()
```
"""
    with open(os.path.join(adapter_dir, "README.md"), "w") as f:
        f.write(model_card)

    api.upload_folder(
        folder_path = adapter_dir,
        repo_id     = repo_id,
        token       = hf_token,
    )
    print(f"  ✅ Pushed → https://huggingface.co/{repo_id}")