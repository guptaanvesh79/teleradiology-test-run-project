# Data dictionary

All data in this package is synthetic. It was generated to resemble the shape of an operational teleradiology worklist, but it describes no real hospital or patient.

## studies.csv (170,076 rows, 2026-06-01 to 2026-09-30)

One row per imaging study that arrived at the reporting service.

| Column | Meaning |
| --- | --- |
| study_id | Unique study identifier |
| arrival_time | When the study arrived, as stored by the source system. The timezone is not documented. |
| hospital_id | Hospital the study came from (H01 to H08) |
| modality | XR, CT, US or MR |
| specialty_required | Subspecialty needed to read the study: NEURO, MSK, BODY, CHEST, or GENERAL (no specific subspecialty) |
| criticality | STAT, URGENT or ROUTINE |
| patient_location | ED (emergency), IP (inpatient) or OP (outpatient) |
| day_total_studies | Number of studies recorded in the source system for that calendar date, across all hours |

Notes:

- Hospitals differ a lot in size. Some hours, and a few whole days, have no rows or very few rows. The files do not say whether this reflects zero demand.
- Arrivals are not spread evenly across the clock as stored.
- The data ends on 2026-09-30. There is no data for October 2026.
