# Wafer Guard — Wafer Anomaly Detector

A leakage-safe repair and audit of the supplied PyTorch supervised autoencoder for the SECOM-derived wafer telemetry dataset.

## Files
- `wafer_guard_fixed.ipynb` — end-to-end audit, preprocessing, training, validation, and test evaluation.
- `AUDIT.md` — findings, corrections, PS-L handling, evaluation protocol, and limitations.
- `secom_telemetry_modified.csv` — supplied dataset.
- `requirements.txt` — Python dependencies.

## Run
Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
jupyter notebook wafer_guard_fixed.ipynb
```

Keep the CSV beside the notebook and run all cells from a clean kernel. The notebook preserves timestamp order, fits preprocessing on training rows only, and reports the actual prevalence of each split. The classifier metrics and the separate invariant-monitor alerts are reported independently. Results are an audit baseline, not production certification.
