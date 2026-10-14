# Titanic Survival Prediction — Machine Learning Pipeline

An end-to-end machine learning project that predicts whether a Titanic passenger survived using passenger and travel-related features. The notebook explores and cleans the data, engineers features, builds reusable preprocessing pipelines, compares classification models, evaluates their performance, and saves the selected Logistic Regression pipeline.

## Project Objective

This is a **binary classification** problem:

- `0` — Did not survive
- `1` — Survived

The goal is to build a repeatable machine learning workflow and evaluate how well the models classify passenger survival.

## Dataset

The project uses the Titanic passenger dataset. The notebook works with features including:

- `Pclass` — passenger class
- `Sex` — passenger gender
- `Age` — passenger age
- `SibSp` — siblings and spouses aboard
- `Parch` — parents and children aboard
- `Fare` — passenger fare
- `Embarked` — port of embarkation
- `Survived` — target label

It also examines `PassengerId`, `Name`, `Ticket`, and `Cabin` during data preparation.

## Workflow

1. Inspect the dataset and explore feature distributions.
2. Check missing values and duplicate records.
3. Treat numerical outliers using the IQR method.
4. Engineer `FamilySize`, `IsAlone`, `HasCabin`, and passenger `Title` features.
5. Impute missing values in `Age`, `Embarked`, and `Fare`.
6. Remove columns not used for model training.
7. Split the data into training and testing sets with stratification.
8. Build a preprocessing pipeline for numerical and categorical features.
9. Train and compare Logistic Regression and Decision Tree classifiers.
10. Evaluate performance and save the selected pipeline.

## Preprocessing and Feature Engineering

- **Age:** imputed using the median grouped by `Sex` and `Pclass`, with the overall median as a fallback.
- **Embarked:** missing values are filled using the mode.
- **Fare:** missing values are filled using the median.
- **Cabin:** represented by a `HasCabin` indicator; the original `Cabin` column is removed.
- **FamilySize:** calculated as `SibSp + Parch + 1`.
- **IsAlone:** indicates whether a passenger travelled alone.
- **Title:** extracted from the passenger name.
- **Numerical features:** scaled with `StandardScaler`.
- **Categorical features:** encoded with `OneHotEncoder(handle_unknown="ignore")`.

The notebook uses a `ColumnTransformer` and combines preprocessing with the classifier in a Scikit-learn `Pipeline`.

## Models and Results

The notebook compares two classifiers using accuracy, precision, recall, F1-score, and confusion matrices.

| Model | Test accuracy |
|---|---:|
| Logistic Regression | **83.24%** |
| Decision Tree | **77.09%** |

Based on the recorded evaluation, the Logistic Regression pipeline is selected. These metrics reflect the results recorded in the notebook; results may differ if the dataset or training workflow changes.

### Recorded Confusion Matrices

**Logistic Regression**

```text
[[97 13]
 [17 52]]
```

**Decision Tree**

```text
[[91 19]
 [22 47]]
```

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Repository Structure

```text
Titanic-ML-Classification/
├── ML-Pipeline on Titanic.ipynb
├── titanic_cleaned.xls
├── titanic_logreg_pipeline.pkl
├── requirements.txt
└── README.md
```

The uploaded cleaned dataset is named `titanic_cleaned.xls`. If you export or save it as a CSV, update the filename accordingly wherever it is referenced.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/Titanic-ML-Classification.git
cd Titanic-ML-Classification
```

Replace `<YOUR_USERNAME>` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you need to install the notebook dependencies manually, use:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `ML-Pipeline on Titanic.ipynb` and run the cells in order. The notebook performs the analysis, preprocessing, training, evaluation, model comparison, and model saving.

## Load the Saved Model

The serialized pipeline can be loaded with Joblib:

```python
import joblib

model = joblib.load("titanic_logreg_pipeline.pkl")
```

The saved pipeline includes the preprocessing steps and trained Logistic Regression model. For predictions, provide new passenger data with the same feature columns and compatible data types expected by the pipeline.

## Key Learnings

- Exploratory data analysis and data quality checks
- Missing-value imputation and outlier treatment
- Feature engineering for tabular data
- Numerical scaling and categorical encoding
- Stratified train-test splitting
- Reusable Scikit-learn preprocessing pipelines
- Logistic Regression and Decision Tree classification
- Classification metrics and confusion matrices
- Saving and reusing a trained model with Joblib

## Future Improvements

- Perform hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
- Use cross-validation for more robust evaluation.
- Add ROC-AUC and precision-recall analysis.
- Compare additional classification algorithms.
- Deploy the saved pipeline through Flask or FastAPI.
- Build a simple web interface for passenger survival predictions.

## Author

**Vamsi Krishna**

Computer Science & Engineering Graduate interested in Machine Learning, Data Science, Python, SQL, and AI/ML.

## License

This project is intended for educational and learning purposes.
