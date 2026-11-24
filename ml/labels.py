"""Single source of truth for the classes. Everyone imports CLASSES from here.

If a class is dropped (cut rule: validation AUC < 0.65), remove it here ONLY.
contracts.py, fusion, the app and the evaluation all follow automatically.
"""
LABEL_MAP = {                       # term in Indiana "Problems"  ->  model pathology name
    "Cardiomegaly": "Cardiomegaly",
    "Pleural Effusion": "Effusion",
    "Pulmonary Atelectasis": "Atelectasis",
    "Opacity": "Lung Opacity",
    "Nodule": "Nodule",
}
CLASSES = list(LABEL_MAP.values())
