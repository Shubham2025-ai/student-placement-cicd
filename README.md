# Student Placement Prediction & Model Quality CI/CD

An ML-based Student Placement Prediction System with a complete CI/CD
pipeline. The pipeline automatically validates the data, runs unit tests,
trains the model, evaluates its accuracy, tests the prediction API, and
deploys the application only when **all** quality gates pass.

## Problem statement

Predict whether a student will be placed based on: CGPA, attendance,
coding score, number of projects, internships, and communication skills.

## CI/CD quality gates

The pipeline (`.github/workflows/ci-cd.yml`) fails automatically if:

| Condition | Enforced by |
|---|---|
| Code quality check fails | `flake8` step |
| Missing/invalid data is detected | `src/data_validation.py` |
| Unit tests fail | `pytest tests/` |
| Required model file is not generated | `test -f models/model.pkl` check after training |
| Accuracy < 80% | `src/evaluate.py` (`ACCURACY_THRESHOLD = 0.80`) |
| Prediction API fails | `scripts/check_api.py` |

The `deploy` job has `needs: build-test-and-validate`, so it only runs if
every gate above passes.

## Project structure

```
student-placement-cicd/
├── app.py                     # Flask prediction API (/health, /predict)
├── generate_data.py           # Creates the synthetic dataset
├── data/
│   └── students.csv           # Dataset
├── src/
│   ├── data_validation.py     # Missing/invalid data checks
│   ├── preprocess.py          # Feature/label split, train/test split
│   ├── train.py                # Trains RandomForestClassifier, saves model
│   └── evaluate.py            # Accuracy gate (fails if < 80%)
├── scripts/
│   └── check_api.py           # CI step that smoke-tests the live API
├── tests/                     # pytest unit tests for every module above
├── models/                    # model.pkl generated here (gitignored)
├── requirements.txt
├── setup.cfg                  # flake8 config
└── .github/workflows/ci-cd.yml
```

## Run it locally

```bash
pip install -r requirements.txt

# 1. (Re)generate the dataset (optional, already committed)
python generate_data.py

# 2. Validate data
python -m src.data_validation

# 3. Run unit tests
pytest tests/ -v

# 4. Train the model
python -m src.train

# 5. Evaluate accuracy (fails if < 80%)
python -m src.evaluate

# 6. Smoke-test the API
python scripts/check_api.py

# 7. Run the API for real
python app.py
# then, in another terminal:
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"cgpa": 8.5, "attendance": 90, "coding_score": 75, "projects": 3, "internships": 1, "communication_score": 80}'
```

## Push to GitHub and watch the pipeline

```bash
git init
git add .
git commit -m "Initial student placement prediction CI/CD pipeline"
git remote add origin https://github.com/<your-username>/student-placement-cicd.git
git branch -M main
git push -u origin main
```

Go to your repo's **Actions** tab — you'll see the `CI - Validate, Test,
Train, Evaluate` job run through every gate, followed by `CD - Deploy
Application` once everything passes.

## Demoing a pipeline failure

To show the pipeline correctly blocking a bad change, pick any one gate:

- **Bad data**: in `generate_data.py`, uncomment the line that injects a
  `NaN` into `cgpa`, regenerate `data/students.csv`, commit, and push.
  The `Validate Data` step will fail and everything after it (including
  deploy) will be skipped.
- **Failing test**: break an assertion in any file under `tests/`.
- **Low accuracy**: temporarily lower `n_estimators`/`max_depth` in
  `src/train.py` to something extreme, or raise `ACCURACY_THRESHOLD` in
  `src/evaluate.py` above what the model can reach.
- **Lint failure**: introduce an unused import or a line over 100 chars.

Then revert the change, commit, and push again to watch the pipeline go
green and deploy.
