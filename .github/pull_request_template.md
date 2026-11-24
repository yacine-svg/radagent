## What this PR does
<!-- one or two lines, e.g. "A2: vision.py + image predictions" -->

## Checklist (AGENTS.md)
- [ ] Branch is up to date with `main`, and `python -m pytest tests/test_contracts.py -q` passes
- [ ] I only edited files I own (AGENTS.md §2), or the other person agreed to this change
- [ ] No contract change, **or** both agreed + `contracts.py` and `contracts.md` updated + `CONTRACT_VERSION` bumped
- [ ] No class names, thresholds or fusion maths hard-coded or copied (imported from `ml.labels` / `contracts`)
- [ ] No secrets, no raw dataset, new dependencies added to `requirements.txt`
- [ ] (A) No test-split tuning, no `findings`/`impression` as input
- [ ] I can explain every line of this PR, including AI-generated code

## How to check it
<!-- command(s) the reviewer runs to see it work -->
