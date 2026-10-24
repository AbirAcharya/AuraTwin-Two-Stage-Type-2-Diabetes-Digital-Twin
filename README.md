# AuraTwin --- Two-Stage Glucose Prediction & Simulation

> An educational Streamlit prototype that combines a patient-profile
> model, meal-nutrition calculations, an illustrative glucose-excursion
> formula, and a lifestyle "what-if" interface.

**Important:** AuraTwin is not clinically validated and must not be used
to diagnose diabetes, guide treatment, or replace professional medical
advice. The app's displayed glucose values and intervention effects are
model outputs/illustrative estimates, not reliable clinical predictions.

## Overview

AuraTwin is a Python web app built with Streamlit. It lets a user enter
a basic health profile, assemble a meal from a food database, enter
recent exercise and hydration information, and view a projected glucose
value and a hypothetical walking/fiber scenario.

The current implementation has three main stages:

1.  **Patient profile:** trains a classifier for the dataset's
    `diabetes` label and a regressor for `blood_glucose_level`, using
    demographic and health-profile features.
2.  **Meal and acute excursion:** calculates meal nutrition from the
    food table, then estimates an acute glucose surge using a manually
    specified formula.
3.  **What-if visualization:** applies fixed, illustrative walking and
    fiber adjustments to the estimated excursion and plots example
    trajectories.

The project is a prototype for educational exploration. Its interface
should not be interpreted as evidence that the underlying calculations
are clinically accurate.

## Features

-   Patient-profile inputs for age, gender, BMI (or height and weight),
    hypertension, heart disease, and smoking history.
-   A HistGradientBoosting classifier that displays the predicted
    probability for the positive `diabetes` class.
-   A HistGradientBoosting regressor that estimates
    `blood_glucose_level` from profile features.
-   A searchable meal builder that allows multiple foods and per-food
    serving quantities.
-   Nutrition summaries for carbohydrates, calories, fiber, weighted
    glycemic index, and estimated glycemic load.
-   Inputs for recent exercise duration, exercise intensity, and
    hydration category.
-   A what-if interface for additional walking and a checkbox for a
    hypothetical soluble-fiber preload.
-   Plotly chart comparing a formula-generated trajectory with an
    intervention-adjusted trajectory.
-   A fallback food table and fallback synthetic cohort when the
    corresponding CSV files are absent.

## Technology stack

-   Python
-   Streamlit
-   pandas
-   NumPy
-   scikit-learn (`HistGradientBoostingClassifier`,
    `HistGradientBoostingRegressor`, `OrdinalEncoder`, `Pipeline`,
    `ColumnTransformer`)
-   Plotly

## Project files

The main application file provided for this README is `main.py`.

The app optionally reads these files from its working directory:

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `pred_food.csv`                     Food nutrition lookup table

  `diabetes_prediction_dataset.csv`   Dataset used to train the
                                      patient-profile classifier and
                                      regressor
  -----------------------------------------------------------------------

Expected columns for `pred_food.csv` include:

-   `Food Name`
-   `Glycemic Index`
-   `Calories`
-   `Carbohydrates`
-   `Protein`
-   `Fat`
-   `Fiber Content`
-   `Suitable for Diabetes`

The code uses the following columns from
`diabetes_prediction_dataset.csv`:

-   `gender`
-   `age`
-   `hypertension`
-   `heart_disease`
-   `smoking_history`
-   `bmi`
-   `HbA1c_level`
-   `blood_glucose_level`
-   `diabetes`

The code currently does not include the dataset files or a separately
saved model artifact in the uploaded `main.py`. If the CSV files are
absent, fallback data is used instead.

## How it works

### Stage 1 --- Patient-profile models

The app selects these features for both models:

-   Categorical: `gender`, `smoking_history`
-   Numeric: `age`, `hypertension`, `heart_disease`, `bmi`

A `ColumnTransformer` applies `OrdinalEncoder` to categorical fields and
passes numeric fields through. Two scikit-learn pipelines are trained
when the app starts:

-   `HistGradientBoostingClassifier`, which predicts the dataset's
    `diabetes` label.
-   `HistGradientBoostingRegressor`, which predicts
    `blood_glucose_level`.

**Implementation note:** although the UI says the classifier is trained
on a "100k cohort," the supplied code does not verify dataset size or
load a bundled 100,000-record dataset. It trains on
`diabetes_prediction_dataset.csv` if present. Otherwise, it generates a
synthetic dataset of 5,000 rows. The fallback synthetic label is derived
from glucose and HbA1c values, while the model inputs omit those two
fields. Predictions from this fallback are therefore not a substitute
for a validated risk model.

The app also calculates an `insulin_resistance_factor` from the
classifier output and BMI using a manually specified formula. This
factor is an application assumption, not a clinically validated
measurement of insulin resistance.

### Stage 2 --- Meal nutrition and estimated excursion

For each selected food, the app multiplies the table's per-serving
carbohydrates, calories, and fiber by the chosen quantity. It then
calculates:

-   Total carbohydrates, calories, and fiber.
-   Carbohydrate-weighted average GI.
-   An estimated meal glycemic load using the GI and carbohydrate values
    in the food table.

The acute glucose surge is then calculated using a fixed formula in
`main.py`:

``` python
raw_surge = (
    (glycemic_load * 1.55) - (total_fiber * 1.8)
) * insulin_resistance_factor * hydration_buff

net_acute_surge = max(0.0, raw_surge - exercise_clearance)
projected_peak_glucose = round(
    stage1_baseline_glucose + net_acute_surge, 1
)
```

The coefficients, exercise adjustment, and hydration adjustment are
manually specified in the application code. The uploaded implementation
does **not** train a separate meal-response model from intervention
data. Treat these values as illustrative prototype calculations, not
validated physiological estimates.

### Stage 3 --- What-if scenario and trajectory

The interface allows the user to choose additional walking minutes and
optionally simulate a soluble-fiber preload. In the supplied code, the
scenario uses fixed assumptions:

-   Walking reduction: up to `0.95` units of estimated excursion per
    selected minute.
-   Fiber attenuation: multiply the excursion by `0.82` when the
    checkbox is selected.
-   The reduction is capped so the adjusted excursion does not fall
    below zero.

The chart then generates a stylized trajectory using a sine-based rising
phase and an exponential decay phase. This trajectory is generated
mathematically; it is not produced by a trained time-series or causal
intervention model.

**Do not describe the current what-if component as a trained
intervention model or a clinically validated counterfactual model.** To
make such a claim, the intervention model, training data, evaluation
procedure, and evidence supporting causal interpretation would need to
be implemented and documented separately.

## Installation

### 1. Clone the repository

Replace the placeholder URL with your GitHub repository URL.

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The uploaded code imports the following packages:

``` bash
pip install streamlit numpy pandas scikit-learn plotly
```

For a reproducible repository, add a tested `requirements.txt` with the
versions used in your environment.

### 4. Add the optional CSV files

Place `pred_food.csv` and/or `diabetes_prediction_dataset.csv` in the
same working directory as `main.py` if you intend to use them. Make sure
you have permission to redistribute any dataset you publish. Do not
commit private or identifiable patient information.

If these files are not present, the app uses its fallback food table and
generated synthetic cohort.

### 5. Run the app

``` bash
streamlit run main.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`.

## Evaluation and reproducibility

The supplied application code does not include a train/test split,
cross-validation, baseline comparison, or reported held-out metrics. As
a result, this README does not claim a measured accuracy, MAE, RMSE, or
other validated performance figure.

Before presenting AuraTwin as an ML portfolio project, consider adding:

1.  A documented dataset source and data dictionary.
2.  Train/validation/test evaluation with a split strategy appropriate
    to the data.
3.  Classification metrics such as precision, recall, F1, ROC-AUC, and a
    confusion matrix where appropriate.
4.  Regression metrics such as MAE and RMSE, plus a simple baseline.
5.  Leakage checks and subgroup/error analysis where appropriate.
6.  Separate validation of any meal-response or intervention model.
7.  Tests for missing values, unknown categories, quantity scaling, and
    scenario calculations.

Metrics should be computed from data that was not used to fit the model.
Do not publish invented or training-only performance metrics as evidence
of generalization.

## Performance

Training records   100,000

ROC-AUC             0.84 (0.837)

Recall at 0.10 threshold   76%

Precision at 0.10 threshold  23%

Inference time   ~3.5 ms

Unique foods   254

Food attributes   12

Trajectory length   120 min

## Limitations and responsible use

-   **Not a medical device:** this prototype has not been clinically
    validated.
-   **Not for diagnosis or treatment:** do not use its output to make
    health decisions.
-   **Heuristic meal-response calculation:** the current acute excursion
    is based on fixed coefficients, not a trained postprandial-response
    model.
-   **Illustrative intervention effects:** walking and fiber scenario
    adjustments use fixed assumptions and do not establish causal
    effects.
-   **Fallback synthetic data:** if the real CSV is missing, the app
    creates a synthetic cohort; this can produce outputs that look
    realistic without representing real-world predictive performance.
-   **Dataset dependence:** results depend on the data supplied and its
    quality, provenance, population, and permitted use.
-   **Nutrition estimates:** food values and serving sizes may vary; GI
    and GL values are estimates and are not individual glucose
    measurements.
-   **No clinical validation of thresholds:** the labels and threshold
    bands in the UI are application choices and should not be
    interpreted as a diagnosis.

Consult a qualified healthcare professional for medical questions.

## Roadmap

Potential improvements:

-   Add a transparent data-source and licensing section.
-   Add a reproducible training script and saved model pipeline.
-   Evaluate the classifier and regressor on held-out data and report
    metrics.
-   Separate model inference from Streamlit presentation code.
-   Add automated tests and input validation.
-   Replace illustrative excursion and intervention assumptions only
    when supported by appropriate data and validation.
-   Add a screenshot or short demo GIF after verifying the app's
    behavior.
-   Add a tested `requirements.txt` and clear version information.

## Contributing

For contributions, open an issue describing the proposed change before
submitting a pull request. Changes that affect predictions should
include tests and a clear explanation of the data, assumptions, and
evaluation.

## License

No license was provided with the uploaded code. Add a `LICENSE` file
before inviting reuse. Until a license is added, do not assume that
others have permission to copy, modify, or redistribute the project.

------------------------------------------------------------------------

Made with Python and Streamlit for educational exploration.
