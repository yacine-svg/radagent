# RadAgent — Data contract v1.0

Agreed on: Tue 24 Nov 2026 · Owners: Person A (Perception & ML) + Person B (Agentic & Platform)

This file is the human-readable version. **`contracts.py` is the executable version**: every rule written here is checked there, and `tests/test_contracts.py` proves it. If this file and `contracts.py` ever disagree, `contracts.py` wins. Fix this file.

```bash
python -m pytest tests/test_contracts.py -q          # 24 passed, 2 skipped
RUN_ML_TESTS=1 python -m pytest tests/test_contracts.py -q   # also checks A's real functions
```

**Change process:** nobody changes a field alone. Agree in the daily sync, edit `contracts.py` + this file in the same pull request, bump `CONTRACT_VERSION`, and make the tests pass.

---

## 1. Global conventions

| Rule | Value |
|---|---|
| Class names and order | Always `CLASSES` from `ml/labels.py`: `Cardiomegaly, Effusion, Atelectasis, Lung Opacity, Nodule`. Never type a class name by hand anywhere else. |
| Per-class lists | Contain **every** class, **in `CLASSES` order**, even when the value is 0 / "absent". |
| Probabilities | Floats in `[0, 1]`, rounded to **4 decimals**. |
| Text signal | `+1` positive · `-1` negated · `0` absent. |
| Paths | Relative to the repo root, forward slashes (`data/demo/x.png`). |
| Heatmap path | `outputs/{case_id}/cam_{label with spaces → _}.png` (use `contracts.heatmap_path()`). |
| `case_id` | `[A-Za-z0-9_-]`, max 64 chars. The router generates `case-<8 hex>` if none is given. Demo cases use fixed ids (`demo-agreement`…). |
| Unknown keys | **Forbidden** (a typo like `probability` instead of `prob` is an error). |
| Errors | **Nodes never raise.** Catch the exception, put a short message in that result's `error` field, and let the critic turn it into a reason (§8). |
| Language | Keys and codes in English. The report (`report_md`) in French. |

---

## 2. Shared maths (owned by the contract, used by both of you)

Defined in `contracts.py` and re-exported by `agents/fusion.py`. **Nobody re-implements them.**

```python
fuse(p_image, s_text, beta)  = sigmoid( logit(clip(p_image, 1e-4, 1-1e-4)) + beta * s_text )
detect_conflict(p_image, s_text):
    "image_pos_text_neg"  if p_image >= 0.70 and s_text == -1
    "text_pos_image_neg"  if p_image <= 0.30 and s_text == +1
    None                  otherwise
in_uncertain_band(p)          = 0.35 <= p <= 0.65
```

`beta` comes from `ml/artifacts/beta.json` (tuned by A on validation). Thresholds come from `ml/artifacts/thresholds.json`.

---

## 3. Input and router

**CaseInput**: what the UI sends to the graph.

| Field | Type | Rule |
|---|---|---|
| `case_id` | str | see §1 |
| `image_path` | str | PNG or JPG |
| `text` | str | may be empty |
| `patient_id` | str \| null | optional |

**RouterFlags** → `state["flags"]`

| Field | Type | Rule |
|---|---|---|
| `has_text` | bool | `= bool(text.strip())` |
| `has_patient` | bool | `= patient_id is not None` |
| `errors` | list[str] | e.g. `"image file not found"` (the image agent will then fail → `image_failed`) |

---

## 4. Vision (Person A → Person B)

### `ml.vision.predict(path)` and MCP tool `classify_image` → **VisionPrediction**
```json
{"findings": [{"label": "Cardiomegaly", "prob": 0.82, "above_threshold": true},
              {"label": "Effusion", "prob": 0.18, "above_threshold": false},
              {"label": "Atelectasis", "prob": 0.22, "above_threshold": false},
              {"label": "Lung Opacity", "prob": 0.31, "above_threshold": false},
              {"label": "Nodule", "prob": 0.06, "above_threshold": false}],
 "model": "densenet121-res224-all + isotonic"}
```
- `prob` is **calibrated** once `calibrators.joblib` exists (raw score before that).
- `above_threshold` = `prob >= thresholds.json["image"][label]` (0.5 before A5).

### `ml.vision.gradcam(path, label, out_png)` → `str` (the path written)

### Image Agent output → `state["image_result"]` = **ImageResult**
VisionPrediction + `heatmaps` + `error`:
```json
{"findings": [...], "model": "...",
 "heatmaps": {"Cardiomegaly": "outputs/demo-agreement/cam_Cardiomegaly.png"},
 "error": null}
```
- Heatmaps for **every class with `above_threshold = true`**, or for the **single top class** if none is above its threshold. No more, no fewer.
- On failure: `{"findings": [], "model": "", "heatmaps": {}, "error": "cannot read image"}`.

---

## 5. Text (Person A → Person B)

### `ml.text_nlp.analyze_text(text)` and `state["text_result"]` → **TextResult**
```json
{"analyzed_text": "62-year-old woman, shortness of breath, history of CHF | Patient history: hypertension; congestive heart failure (2019)",
 "entities": [{"text": "shortness of breath", "type": "symptom", "negated": false, "span": [19, 38]},
              {"text": "history of CHF", "type": "history", "negated": false, "span": [40, 54]},
              {"text": "congestive heart failure", "type": "history", "negated": false, "span": [88, 112]}],
 "suspected": {"Cardiomegaly": "positive", "Effusion": "absent", "Atelectasis": "absent",
               "Lung Opacity": "absent", "Nodule": "absent"},
 "history_used": true,
 "error": null}
```
| Field | Rule |
|---|---|
| `analyzed_text` | The **exact** string the spans refer to: the cleaned text (XXXX removed, spaces collapsed), plus history if used. The UI highlights entities **in this string**, not in the raw input. |
| `entities[].span` | `[start, end)`, and `analyzed_text[start:end] == text` **exactly**. |
| `entities[].type` | `symptom` · `history` · `finding` |
| `suspected` | One key per class (all classes), value `positive` · `negated` · `absent`. |
| `history_used` | `analyze_text` always returns `false`. The **Text Agent** sets `true` when it appended patient history. |
| History format | Text Agent builds `text + " | Patient history: " + "; ".join(history)` (constant `HISTORY_SEPARATOR`) **before** calling `analyze_text`. |
| Empty text | `analyzed_text: ""`, `entities: []`, all classes `absent`. |
| On failure | Same as empty text + `"error": "..."`. Fusion then uses `s_text = 0` for every class. |

---

## 6. MCP tools (Person B), exact signatures

| Server | Tool | Arguments | Returns |
|---|---|---|---|
| vision-server | `classify_image` | `image_path: str` | VisionPrediction |
| vision-server | `gradcam` | `image_path: str, label: str, case_id: str` | `str`: `heatmap_path(case_id, label)` |
| patient-server | `get_patient_history` | `patient_id: str` | PatientRecord (never raises; `found: false` if unknown) |
| knowledge-server | `search_pubmed` | `query: str, max_results: int = 3` | list[Reference] |
| knowledge-server | `icd10_lookup` | `condition: str` (a class name) | ICD10 (`code: null` if unknown) |
| knowledge-server | `guideline_snippets` | `condition: str, k: int = 2` | list[GuidelineSnippet] |

FastMCP sends results as JSON. On the client side, if a tool result arrives as a string, `json.loads` it, then validate it with the model above.

**PatientRecord**: SYNTHETIC data only
```json
{"patient_id": "P007", "found": true, "age": 62, "sex": "F", "smoker": false,
 "history": ["hypertension", "congestive heart failure (2019)"],
 "prior_studies": ["Chest X-ray 2023: enlarged cardiac silhouette"]}
```
Not found: `{"patient_id": "P999", "found": false}` (other fields default to null / []).

**ICD10**: `{"condition": "Cardiomegaly", "code": "I51.7", "label": "Cardiomegaly"}`
Codes used: Cardiomegaly I51.7 · Effusion J90 · Atelectasis J98.11 · Lung Opacity R91.8 · Nodule R91.1 (check each once on an official ICD-10 browser).

**Reference**: `{"pmid": "12345678", "title": "...", "journal": "...", "year": "2021"}` (`pmid` is digits only, `year` is 4 digits or "").

**GuidelineSnippet**: `{"text": "...written in your own words...", "source": "https://..."}`

---

## 7. Knowledge Agent → `state["knowledge"]` = **KnowledgeResult**

Evidence is **prefetched for every class** (5 classes, all cached), which is why this agent can run in parallel with the others.
```json
{"by_class": {
   "Cardiomegaly": {"icd10": {...ICD10}, "refs": [{...Reference}, {...}], "guideline": {...GuidelineSnippet} | null},
   "Effusion": {...}, "Atelectasis": {...}, "Lung Opacity": {...}, "Nodule": {...}},
 "error": null}
```
- `by_class` keys = all classes, in order. On failure: `{"by_class": {}, "error": "..."}`.

---

## 8. Fusion → `state["fused"]` = **FusedResult**

```json
{"findings": [
   {"label": "Cardiomegaly", "p_image": 0.82, "s_text": 1, "p_fused": 0.9253,
    "threshold": 0.48, "above_threshold": true, "conflict": null},
   {"label": "Effusion", "p_image": 0.90, "s_text": -1, "p_fused": 0.768,
    "threshold": 0.42, "above_threshold": true,
    "conflict": {"type": "image_pos_text_neg",
                 "message": "Image suggests Effusion (p=0.90) but the clinical text says it is absent."}},
   "... one entry per class, in CLASSES order ..."],
 "beta": 1.0}
```
| Field | Rule (checked) |
|---|---|
| `p_image` | = `image_result` prob for that class |
| `s_text` | = signal of `text_result.suspected` (0 everywhere if the text failed) |
| `p_fused` | = `fuse(p_image, s_text, beta)` (±0.001) |
| `threshold` | = `thresholds.json["fused"][label]` |
| `above_threshold` | = `p_fused >= threshold` |
| `conflict` | present **exactly** when `detect_conflict(p_image, s_text)` is not None, with that `type` |
| whole object | `null` only when the image failed |

---

## 9. Critic → `state["decision"]` + `state["reasons"]`

**Reason**: `{"code": "...", "severity": "...", "label": "Effusion" | null, "message": "..."}`

| Code | Severity | When (all mandatory) |
|---|---|---|
| `uncertain_conflict` | **abstain** | a class has a conflict **and** its `p_fused` is in [0.35, 0.65] |
| `all_uncertain` | **abstain** | every class has `p_fused` in [0.35, 0.65] |
| `image_failed` | **abstain** | `image_result.error` is set |
| `conflict` | warning | a class has a conflict and its `p_fused` is outside the band |
| `text_failed` | warning | `text_result.error` is set |
| `knowledge_failed` | warning | `knowledge.error` is set |
| `patient_not_found` | info | a `patient_id` was given but `found` is false |
| `no_text` | info | `has_text` is false |
| `entity_dropped` | info | optional: the critic removed an entity not found verbatim |

`decision` = `"abstain"` **exactly when** at least one reason has severity `abstain`, otherwise `"ok"`.
The `abstain` node then runs (and appears in the trace), and the explainer still writes a report explaining why.

---

## 10. Explainer → `state["report_md"]` + `state["citations"]`

- Markdown in French with **these 6 headings, in this order**:
  `## Résumé` · `## Observations image` · `## Éléments cliniques` · `## Points de vigilance` · `## Références` · `## Avertissement`
- References are written as `PMID: 12345678`.
- `citations` = exactly the set of PMIDs that appear in `report_md`, and each one must come from `knowledge` (no invented references).
- The `## Avertissement` section contains this sentence **verbatim** (constant `DISCLAIMER`):
  > Aide à l'analyse uniquement — ne constitue pas un diagnostic médical.
- Only findings with `above_threshold = true` in `fused` are presented as "probable". Every conflict from `reasons` appears under *Points de vigilance*.

---

## 11. Trace → `state["trace"]` (every node appends; LangGraph reducer `operator.add`)

```json
{"node": "image_agent", "message": "5 findings scored, Grad-CAM generated",
 "duration_ms": 1840, "tool": "vision-server.classify_image"}
```
- `node` ∈ `router, image_agent, text_agent, knowledge_agent, fusion, critic, abstain, explainer`
- The first event is `router` and the last is `explainer`. `abstain` appears only when `decision == "abstain"`.
- The UI shows `event["message"]` (in the Streamlit snippet from the plan, use `[-1]["message"]`).

---

## 12. Final state = **FinalState** (what the UI renders)

Keys: `contract_version, case_id, image_path, text, patient_id, flags, image_result, text_result, patient, knowledge, fused, decision, reasons, report_md, citations, trace`.
`patient` is `null` when no `patient_id` was given. Full examples are in `examples/`:

| File | What it shows |
|---|---|
| `case_agreement.json` | Image and text (with patient history) agree on cardiomegaly → `ok` |
| `case_conflict_warning.json` | Image 0.90 effusion, text says "no effusion" → conflict, fused 0.77 → `ok` + warning |
| `case_abstain.json` | Same conflict but fused 0.54 (uncertain) + unknown patient → `abstain` |

The PMIDs in the examples (`9…`) are placeholders, not real articles.

---

## 13. Files shared between A and B

| File | Writer | Format |
|---|---|---|
| `data/splits.csv` | A | `uid, filename, split (val\|test), <one 0/1 column per class>` |
| `data/preds_image.csv` | A | `uid, filename, split, indication, <class> (0/1), <class>_score (raw float)` |
| `data/text_cache.json` | A | `{ cleaned_text: TextResult }` |
| `ml/artifacts/calibrators.joblib` | A | `dict[class → sklearn IsotonicRegression]` |
| `ml/artifacts/thresholds.json` | A | `{"image": {class: float}, "fused": {class: float}}` |
| `ml/artifacts/beta.json` | A | `{"beta": float}` |
| `results/test_preds.csv` | A | preds_image columns + `<class>_s, <class>_img, <class>_txt, <class>_fus` |
| `data/icd10.csv` | B | `condition, code, label` |
| `data/patients.db` | B | SQLite, table `patients(patient_id, age, sex, smoker, history, prior_studies)`, list fields stored as JSON strings |
| `data/cache/pubmed.json` | B | `{ query: list[Reference] }` |

---

## 14. Changes vs. the earlier plan PDFs (apply these when you write the code)

1. **`analyze_text` adds `"analyzed_text": text`** to its result (both the empty-text return and the normal return). One line in each place.
2. **History separator** is `" | Patient history: "`, not a newline (the cleaning step would collapse a newline anyway).
3. **`gradcam` MCP tool takes `case_id`** and writes to `heatmap_path(case_id, label)`.
4. **Knowledge result is wrapped** in `{"by_class": {...}, "error": null}`.
5. **Fused findings include `threshold` and `above_threshold`**, and the formula comes from `contracts.fuse` (`agents/fusion.py` re-exports it, so A's `from agents.fusion import fuse` works from day 1).
6. **Reasons are objects** (`code, severity, label, message`), not strings.
7. **Trace events are objects** (`node, message, duration_ms, tool`), not strings.
8. **Explainer returns `citations`** and must use the 6 fixed headings + the disclaimer.
