"""Create data/patients.db: 20 SYNTHETIC patients (invented for the demo, not real people).

Run:  python -m data.make_patients      (patient-server also runs it automatically if the db is missing)
Table: patients(patient_id, age, sex, smoker, history, prior_studies); lists stored as JSON strings.
"""
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "patients.db"

# (id, age, sex, smoker, history, prior studies) — all fictional
PATIENTS = [
    ("P001", 45, "M", False, ["no significant history"], []),
    ("P002", 71, "F", False, ["hypertension", "type 2 diabetes"], ["Chest X-ray 2024: normal"]),
    ("P003", 58, "M", True, ["COPD", "30 pack-years"], ["Chest X-ray 2022: hyperinflation"]),
    ("P004", 66, "F", False, ["breast cancer treated 2018"], ["CT chest 2023: no metastasis"]),
    ("P005", 39, "F", False, ["asthma"], []),
    ("P006", 80, "M", False, ["atrial fibrillation", "chronic kidney disease"], ["Chest X-ray 2025: small bilateral effusions"]),
    ("P007", 62, "F", False, ["hypertension", "congestive heart failure (2019)"], ["Chest X-ray 2023: enlarged cardiac silhouette"]),
    ("P008", 55, "M", True, ["smoker", "weight loss for 3 months"], []),
    ("P009", 73, "M", False, ["abdominal surgery 5 days ago"], []),
    ("P010", 28, "M", False, ["no significant history"], []),
    ("P011", 64, "F", True, ["lung nodule under follow-up (8 mm, 2025)"], ["CT chest 2025: 8 mm solid nodule right upper lobe"]),
    ("P012", 50, "M", False, ["treated tuberculosis (2010)"], ["Chest X-ray 2020: calcified granuloma left mid zone"]),
    ("P013", 77, "F", False, ["dilated cardiomyopathy"], ["Echocardiography 2024: reduced ejection fraction"]),
    ("P014", 42, "F", False, ["pneumonia treated 6 weeks ago"], ["Chest X-ray 6 weeks ago: right lower lobe consolidation"]),
    ("P015", 68, "M", True, ["COPD", "recurrent bronchitis"], []),
    ("P016", 59, "F", False, ["rheumatoid arthritis"], []),
    ("P017", 83, "M", False, ["heart failure", "aortic stenosis"], ["Chest X-ray 2024: cardiomegaly"]),
    ("P018", 35, "F", False, ["pregnancy 20 weeks"], []),
    ("P019", 61, "M", False, ["lymphoma in remission"], ["CT chest 2024: no active disease"]),
    ("P020", 47, "M", True, ["smoker"], []),
]


def build(db: Path = DB) -> Path:
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE patients (patient_id TEXT PRIMARY KEY, age INTEGER, sex TEXT, "
                    "smoker INTEGER, history TEXT, prior_studies TEXT)")
        con.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?)",
                        [(pid, age, sex, int(smoker), json.dumps(h), json.dumps(p))
                         for pid, age, sex, smoker, h, p in PATIENTS])
    return db


if __name__ == "__main__":
    print("wrote", build(), "with", len(PATIENTS), "synthetic patients")
