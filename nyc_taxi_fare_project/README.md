# NYC Taxi Upfront Fare Estimation (Deliverable 2)

Real-World AI-Driven Analytics Solution Using Python. This deliverable covers model optimization, real-world impact and ethical evaluation.

| File | Purpose |
|---|---|
| `REPORT.md` | Written deliverable: hyperparameter tuning, final evaluation, ethics, real-world application, conclusion |
| `NYC_Taxi_Fare_Deliverable2.ipynb` | Fully executed notebook with all code, tables and figures |
| `build_notebook.py` | Script that regenerates the notebook from source |
| `figures/` | Figures used in the report |

## Run it
**Google Colab:** upload the notebook and choose *Runtime → Run all* (about 7 minutes).

**Locally:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow jupyter
jupyter nbconvert --to notebook --execute --inplace NYC_Taxi_Fare_Deliverable2.ipynb
```

By default the notebook uses the 6,433-trip TLC sample from March 2019 (`DATA_SOURCE = "sample"`); this is the run reported in `REPORT.md`. For a full-scale run in Colab, set `DATA_SOURCE = "tlc"` in Part 0. The notebook then downloads the official TLC yellow and green parquet files for `TLC_MONTH` and samples `TLC_SAMPLE_ROWS` rows from them.
