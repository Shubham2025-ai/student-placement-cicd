"""
Generates a synthetic student placement dataset and saves it to data/students.csv.

Run this once to (re)create the dataset:
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 400

cgpa = np.round(np.random.uniform(5.0, 10.0, N), 2)
attendance = np.round(np.random.uniform(50, 100, N), 1)
coding_score = np.round(np.random.uniform(30, 100, N), 1)
projects = np.random.randint(0, 6, N)
internships = np.random.randint(0, 4, N)
communication_score = np.round(np.random.uniform(30, 100, N), 1)

# Weighted "true" placement score, then add noise so the problem isn't trivially
# linear -- this keeps the model's accuracy in a realistic (but comfortably >80%)
# range instead of ~100%.
score = (
    0.30 * (cgpa / 10)
    + 0.20 * (attendance / 100)
    + 0.20 * (coding_score / 100)
    + 0.10 * (projects / 5)
    + 0.10 * (internships / 3)
    + 0.10 * (communication_score / 100)
)
noise = np.random.normal(0, 0.05, N)
score_noisy = score + noise

placed = (score_noisy >= np.median(score_noisy)).astype(int)

df = pd.DataFrame(
    {
        "cgpa": cgpa,
        "attendance": attendance,
        "coding_score": coding_score,
        "projects": projects,
        "internships": internships,
        "communication_score": communication_score,
        "placed": placed,
    }
)

# Inject a few missing values on purpose so data_validation.py has something
# real to catch when you want to demo a CI failure.
# (Left commented out by default -- uncomment to test the "bad data" path.)
# df.loc[3, "cgpa"] = np.nan

df.to_csv("data/students.csv", index=False)
print(f"Wrote data/students.csv with {len(df)} rows")
print(df["placed"].value_counts())
