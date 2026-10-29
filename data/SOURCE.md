# Dataset source and reuse

- **Dataset:** Daily Demand Forecasting Orders, UCI Machine Learning Repository, dataset 409.
- **Source page:** https://archive.ics.uci.edu/dataset/409/daily+demand+forecasting+orders
- **Direct no-login download:** https://archive.ics.uci.edu/static/public/409/daily%2Bdemand%2Bforecasting%2Borders.zip
- **Attribution:** Ferreira, R., Martiniano, A., Ferreira, A., Ferreira, A., & Sassi, R. (2016). Daily Demand Forecasting Orders [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BC8T
- **License:** Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- **Original file:** Daily_Demand_Forecasting_Orders.csv; source bytes are unchanged. The bundled file permits offline execution. A missing original can be downloaded automatically; a changed checksum causes an explicit error.
- **Original SHA-256:** `adf8dda021beae76b13b1cbb5b3c075d064f03b3353a711c1ddc357c6f89e924`

UCI describes this as real data from a Brazilian logistics company, collected over 60 days, and labels it time-series data. The CSV has 60 observations and 13 columns but no year, month identifier or exact dates. We preserve its row order as an explicit sequence assumption; this does not independently verify calendar chronology.

## Transformations and audit observations

- Read the semicolon-separated file with periods interpreted as thousands separators. `539.577` becomes 539577 orders. This interpretation is supported by integer metadata and exact component-sum checks across all 60 rows.
- Target totals exactly equal both A+B+C order-type counts and non-urgent+urgent+fiscal counts. These same-period components are not used as forecasting inputs.
- The source weekday codes are 2=Monday through 6=Friday. Transitions before observations 33 and 40 break the consecutive Monday-Friday cycle. We do not invent dates, impute missing observations, sort by repeating calendar fields or label lag 5 as an assured same-weekday value.
- Week-of-month decreases at observations 20 and 40, consistent with resets but insufficient to assign actual months/years.
- `cleaned_demand.csv` retains the target and two calendar-code columns with descriptive names and adds a one-based source observation ID. Calendar fields remain audit context only.
- `outputs/lagged_features.csv` adds three strictly historical lags and a shifted three-observation mean; these are project transformations, not source columns.

No synthetic values, fabricated dates, interpolated observations or randomized row splits are introduced.
