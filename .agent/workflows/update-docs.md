---
description: Update project documentation (README, ADRs, roadmap) after implementation changes
---

# Update Documentation

After any implementation change, update the following docs to stay in sync:

1. Check what files changed since last commit:
```bash
git diff --name-only HEAD~1
```

2. If `training/nlhe_trainer.py` or `scripts/train.py` changed:
   - Update CLI options in `README.md` under "Full CLI Options"
   - Update training examples in `README.md` under "Training"
   - Verify all `--flag` references match actual argparse definitions

3. If `evaluation/evaluator.py` or `scripts/evaluate.py` changed:
   - Update evaluation examples in `README.md` under "Evaluation"

4. If any model files changed (`model/*.py`):
   - Update architecture description in `README.md`
   - Update the file tree if new files were added

5. If `training/personality.py` changed:
   - Update personality/curriculum docs in README

6. Verify README accuracy:
// turbo
```bash
python3 scripts/train.py --help
```
// turbo
```bash
python3 scripts/evaluate.py --help
```
