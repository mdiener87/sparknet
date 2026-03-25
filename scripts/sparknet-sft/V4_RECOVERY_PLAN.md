# SparkNet-400M SFT v4 Recovery Plan

## Goal

Produce a small but coherent assistant from `sparknet-400m-v1`.

The target is not deep reasoning. The target is stable instruction following,
reasonable short answers, and usable multi-turn chat behavior.

## What v4 changes

1. Keep the `sft_chat_v3` conversation format for training.
   It avoids cross-conversation packing and keeps the newest turns intact.

2. Train with a real held-out split by default.
   `train_sft.py` now uses `eval_fraction=0.02` and evaluates every 100 steps.

3. Select checkpoints by behavior, not only by loss.
   Every eval now runs a fixed prompt suite from
   `configs/sparknet-400m/sft_eval_prompts_v4.json` and writes generations to
   `sample_generations.jsonl` in the checkpoint output directory.

4. Train long enough to matter.
   The default `v4` budget is `300M` tokens, which is about `1.5` passes over
   the current `195,316`-row SFT dataset with the existing batch settings.

5. Make future dataset rebuilds cleaner.
   `build_dataset.py` now defaults to `datasets/sft_chat_v4`, stores the correct
   `attention_mask`, and slightly increases OASST representation.

## Suggested workflow

1. Run `scripts/sparknet-sft/run_sft.sh`.
2. Watch `logs/sparknet-400m-v1-instruct-v4/train.log`.
3. Compare generations across eval checkpoints.
4. Keep the checkpoint that is most coherent and least repetitive, even if it is
   not the final checkpoint.

## Files

- Training config: `configs/sparknet-400m/sft_v4.json`
- Eval prompt suite: `configs/sparknet-400m/sft_eval_prompts_v4.json`
- Trainer: `scripts/sparknet-sft/train_sft.py`
- Launcher: `scripts/sparknet-sft/run_sft.sh`
- Dataset builder: `scripts/sparknet-sft/build_dataset.py`
