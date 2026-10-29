# Verification record - Project 3

- Ran the complete original CSV -> audited integer counts -> past-only lag features -> fixed training fit -> holdout comparison -> CSV/JSON/chart export workflow successfully.
- Environment: Windows, Python 3.12, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1 and matplotlib 3.11.2. The shared environment's dependency check reported no broken requirements.
- Last measured application runtime: **0.761 seconds**, including source loading, auditing, model fitting, predictions and chart/export work. Python startup and top-level imports are outside this timer. Runtime varies by laptop.
- Original source hash matches `adf8dda021beae76b13b1cbb5b3c075d064f03b3353a711c1ddc357c6f89e924`. Source bytes were downloaded directly from the official UCI archive and are bundled with attribution.
- Correctly parsed periods as thousands separators. First target is **539577**, not 539.577; both independent same-period component identities hold for all 60 rows.
- Calendar audit identified weekday-cycle breaks before observations **33 and 40** and week-field decreases at **20 and 40**. No exact dates, missing observations or confirmed consecutive-business-day chronology were invented.
- Source-order split: **47 training labels at observations 4-50**; **10 holdout labels at 51-60**. Only lag_1, lag_2 and lag_3 enter the regression. Same-period order components and calendar labels do not enter it.
- Holdout MAE / RMSE in orders: last observation **44572.30 / 56989.24**; regression **46172.71 / 59811.13**; trailing-three mean **50305.50 / 61185.46**. Regression did not outperform the simplest baseline.
- Six standard-library unittest checks passed in **0.105 seconds**: source units/calendar boundaries, modified-source rejection, shifted lag/mean arithmetic, current/future-target perturbation, split/independent scoring and next-observation inputs.
- Perturbing all labels from observation 51 onward left the training coefficients and observation-51 forecast unchanged. Later holdout lag inputs may use earlier actual holdout labels only after those observations become available.
- Independently recomputed MAE and RMSE from exported prediction arrays in the checks; they match scikit-learn's results.
- Re-ran from the portfolio root to check independence from the working directory. All **six** generated CSV checksums and model scores were unchanged; only measured runtime metadata changed.
- Both saved figures were visually checked for readable axes, labels and clearly marked holdout boundaries. README file links were checked against saved artifacts.
- The three observation-61 forecasts are unscored illustrations. Their true outcome, date and prediction errors are unknown; no accuracy or interval is claimed for them.
- No hosted deployment, paid API, GPU, random split or other project import is required.

This verifies a small source-order demonstration, not the missing calendar chronology, production readiness or real-world operational improvement.
