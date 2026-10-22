# C2 Prediction–Reality Divergence: FastWAM reference adapter

This adapter feeds predicted RGB frames back into subsequent policy inputs.
Its evaluation code, feedback logic, and recording hooks are bundled in this
release. FastWAM model code and checkpoints remain external dependencies.

Run through the public entry point:

```bash
python3 scripts/run_eval.py --setting C2.recursive_self_prediction.all_steps \
  --model fastwam --suite libero_10 --task 2 --trials 1 --gpu 0
```

An adapter for this setting must expose predicted RGB frames.
