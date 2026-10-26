# Assignment 3 – Bayesian Inference for Heart Disease Diagnosis

| File | What it is |
|---|---|
| `Bayesian_Heart_Disease_Report.docx` | Final report (APA references, 14 figures, 8 tables) |
| `Bayesian_Heart_Disease_Diagnosis.ipynb` | Executed notebook: preprocessing, priors, PyMC models, validation, refinement, sequential updating |
| `bayes_heart.py` | Commented helper module used by the notebook |
| `data/` | Processed UCI Heart Disease files (Cleveland, Hungary, Switzerland, Long Beach VA) |
| `figures/`, `results/` | Every figure and table the report uses, written by the notebook |

Reproduce:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Bayesian_Heart_Disease_Diagnosis.ipynb
```

Runtime is about 6 minutes on 4 cores (mostly the 5-fold cross-validation).
