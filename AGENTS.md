# RadAgent — Team rules (for both of us AND for any AI assistant we use)

> **If you are an AI assistant** (Cursor, ChatGPT, Claude, Copilot…) helping on this repository:
> these rules override your defaults. Follow them exactly. If a request would break one of them,
> **do not do it**: say which rule it breaks and propose what the human should discuss with their partner.

## 1. The project in 5 lines

RadAgent is an MSc prototype (Thème 3): an agentic **decision-support** system for chest X-rays, not a diagnostic tool.
An image model (DenseNet-121, torchxrayvision) and a text analyser (LLM via Groq) produce evidence. Agents (LangGraph) collect extra evidence through 3 MCP servers, fuse everything, check it with a critic (which may abstain), and write an explained report shown in a Streamlit app.
Two people, two weeks (24 Nov → 8 Dec 2026). Plan: `docs/RadAgent_2-week_plan.pdf`.

## 2. Who owns what

| Role | Person | Owns (only this person edits these files) |
|---|---|---|
| **A — Perception & ML** | partner | `ml/vision.py`, `ml/text_nlp.py`, `ml/artifacts/`, `eval/`, `scripts/`, `data/splits.csv`, `data/preds_image.csv`, `data/text_cache.json`, `data/text_gold.csv`, `data/demo/`, `results/`, `report/figures/` |
| **B — Agentic & Platform** | yacine-svg | `agents/` (except the shared maths in `contracts.py`), `servers/`, `app/`, `data/icd10.csv`, `data/patients.db`, `data/make_patients.py`, `data/guidelines/`, `data/cache/` |
| **Shared: change only when BOTH agree** | both | `contracts.py`, `contracts.md`, `ml/labels.py`, `examples/`, `tests/test_contracts.py`, `AGENTS.md`, `README.md`, `requirements.txt`, `.github/` |

**Rule 2.1:** never edit a file owned by the other person. If you need something from their file, ask them.
**Rule 2.2:** shared files change only in a pull request that the other person approves.

## 3. The contract is the law

The source of truth, in this order:
1. **`contracts.py`**: executable rules (pydantic models, shared maths, constants)
2. **`contracts.md`**: the same rules, readable
3. `docs/*.pdf`: plans and guides. If a PDF disagrees with the contract, **the contract wins**.

**Rule 3.1:** every output exchanged between components must validate against its model in `contracts.py` (`VisionPrediction`, `ImageResult`, `TextResult`, `PatientRecord`, `ICD10`, `Reference`, `GuidelineSnippet`, `KnowledgeResult`, `FusedResult`, `Reason`, `TraceEvent`, `FinalState`).
**Rule 3.2:** never change a field name, type, class name, constant or rule in your own code to "make it work". Fix your code to match the contract. If the contract itself seems wrong, stop and discuss it. A contract change = both agree + `contracts.py` and `contracts.md` updated in the same PR + `CONTRACT_VERSION` bumped + tests pass.
**Rule 3.3:** class names come **only** from `from ml.labels import CLASSES`. Never type `"Cardiomegaly"` etc. as a list anywhere else.
**Rule 3.4:** the fusion maths and thresholds come **only** from the contract: `from agents.fusion import fuse, detect_conflict, in_uncertain_band` and the constants `CONFLICT_HIGH`, `CONFLICT_LOW`, `UNCERTAIN_BAND`, `HISTORY_SEPARATOR`, `DISCLAIMER` from `contracts`. **Never re-implement or copy them.**

## 4. Public interfaces: do not rename, do not change signatures

```python
# Person A — imported by Person B
ml.vision.predict(path: str) -> dict                       # contracts.VisionPrediction
ml.vision.gradcam(path: str, label: str, out_png: str) -> str   # returns the path written
ml.text_nlp.clean(text) -> str
ml.text_nlp.analyze_text(text: str, model: str | None = None) -> dict   # contracts.TextResult (includes analyzed_text)
ml.text_nlp.to_signal(suspected: dict) -> dict             # {class: -1|0|1}

# Person B — MCP tools (exact names and arguments)
vision-server:    classify_image(image_path) · gradcam(image_path, label, case_id)
patient-server:   get_patient_history(patient_id)
knowledge-server: search_pubmed(query, max_results=3) · icd10_lookup(condition) · guideline_snippets(condition, k=2)

# Shared files (formats in contracts.md §13)
ml/artifacts/thresholds.json  {"image": {class: float}, "fused": {class: float}}
ml/artifacts/beta.json        {"beta": float}
ml/artifacts/calibrators.joblib   dict[class -> IsotonicRegression]
```

## 5. Science rules (Person A + any AI doing ML work)

- **5.1 No leakage:** the text input is the `indication` column only. **Never** use `findings` or `impression` as model or evaluation input.
- **5.2 Splits are fixed:** `data/splits.csv` (30% val / 70% test, split by `uid`, `random_state=42`). Never re-split.
- **5.3 Tune on validation only:** calibrators, thresholds, β, prompt changes, class choice. **The test split is used once**, in A6, and its numbers are reported as they come out.
- **5.4 Report honestly:** never invent, round up or "improve" a number. If text doesn't help (β = 0), say so.
- **5.5 Reproducible:** every number in the report must come from `python -m eval.run_eval` / `python -m eval.make_figures`.
- **5.6 Medical safety:** the system is decision support only. The disclaimer stays in every report. No feature may present a result as a diagnosis.

## 6. Engineering rules (both + any AI)

- **6.1 Nodes and tools never raise:** catch the exception and put a message in the result's `error` field (contract §1).
- **6.2 Probabilities** are floats in [0, 1], rounded to 4 decimals. Paths are relative to the repo root with forward slashes.
- **6.3 No secrets in code or git:** API keys only in `.env` (ignored) or Kaggle Secrets. Never print or paste them.
- **6.4 No raw dataset in git** (`data/raw/` is ignored). Only small derived files (CSV, JSON, a few demo PNGs).
- **6.5 New dependency** → add it to `requirements.txt` in the same PR and tell the other person.
- **6.6 Keep it simple:** no new frameworks, no new services, no refactors of working code two weeks before delivery. Prefer the smallest change that works.
- **6.7 AI-generated code** must be read and understood before committing. You must be able to explain every line in front of the jury.

## 7. Git workflow

- **Never commit directly to `main`.** One branch per block: `a2-vision`, `a3-text`, `b1-skeleton`, `b2-knowledge-server`…
- Commit messages: `A2: vision.py + image predictions`, `B1: graph skeleton with fake agents`.
- Before opening a pull request:
  ```bash
  git checkout main && git pull && git checkout <your-branch> && git merge main
  python -m pytest tests/test_contracts.py -q          # must pass
  ```
- The other person skims every pull request (5 minutes) before it is merged. GitHub Actions runs the contract tests automatically on every PR; a red ❌ means do not merge.
- `main` must always run.

## 8. Communication

- Daily 15-minute sync: what runs (show it), what's next, what blocks you.
- Anything that touches a shared file, the contract, the class list, the model names in `.env` or the schedule is decided **together**, never alone.
- If you're late on your block, say so at the sync. The cut rules are in the plan (§2.5): drop patient-server → re-check loop → guideline RAG, in that order. Never cut the core.

## 9. Quick commands

```bash
python -m pytest tests/test_contracts.py -q                     # contract tests (always)
RUN_ML_TESTS=1 python -m pytest tests/test_contracts.py -q      # + A's real functions (macOS/Linux)
$env:RUN_ML_TESTS="1"; python -m pytest tests/test_contracts.py -q   # same, Windows PowerShell
python -m eval.calibrate && python -m eval.run_eval              # A: validation fit, then test numbers
python -m agents.graph                                           # B: run the graph on an example case
```
