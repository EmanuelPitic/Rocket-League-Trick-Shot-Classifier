# Rocket League Trick Shot Classifier

Time-series classification model for identifying different types of trick shots in Rocket League gameplay using machine learning.

## Project Overview

This project was developed as part of the **Introduction to Artificial Intelligence (194.025WS)** course at TU Wien. The goal is to classify Rocket League trick shots based on in-game telemetry data including player inputs, ball physics, and spatial positioning.

## Results

- **Best Model Accuracy:** 85.4% (5-fold stratified cross-validation)
- **Baseline (Decision Tree):** 72.4%
- **Feature Count:** 46 engineered features → 23 selected features (50% reduction)
- **Model:** Random Forest with automated feature selection

## Methodology

### 1. Feature Engineering
Since the raw data consists of time-series recordings, we transformed sequential data into static feature vectors using:

- **Statistical Aggregations:** Mean, std, min, max for continuous variables (BallAcceleration, PlayerSpeed, etc.)
- **Temporal Features:** First/last values and deltas to capture state changes
- **Input Summaries:** Aggregated player inputs (jump, boost, slide frequency)

This resulted in **46 engineered features** from the raw time-series data.

### 2. Feature Selection
- Used `SelectFromModel` with Random Forest to identify the most predictive features
- Applied median threshold to reduce features by ~50%
- Top features included: `DistanceCeil_min`, `BallAcceleration_max`, `DistanceBall_first`

### 3. Model Selection & Hyperparameter Tuning
- **GridSearchCV** with stratified 5-fold cross-validation
- Optimized parameters:
  - `n_estimators`: [100, 200]
  - `max_depth`: [5, 10, 15, None]
  - `min_samples_leaf`: [1, 2, 4]
- **Best Configuration:** 100 trees, max_depth=10, min_samples_leaf=1

### 4. Class Imbalance Handling
- Used `class_weight='balanced'` to handle imbalanced trick shot distributions
- Stratified sampling in cross-validation to preserve class distributions

## Repository Structure

```
Rocket-League-Trick-Shot-Classifier/
├── notebook_for_submission.ipynb    # Main notebook with full implementation
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib

### Running the Notebook
1. Ensure the dataset is in the correct directory structure:
   ```
   194-025-ws-eml-project-competition/
   ├── rocketskillshots_train.csv
   └── rocketskillshots_test.csv
   ```
2. Open `notebook_for_submission.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially

## Key Insights

1. **Distance-based features** (ceiling, wall, ball proximity) were most predictive
2. **Ball acceleration patterns** distinguished between trick shot types
3. Feature selection improved generalization by reducing overfitting
4. Random Forest outperformed Decision Tree baseline by **13.2%**

## Model Performance

| Model | Cross-Validation Accuracy |
|-------|---------------------------|
| Decision Tree (Baseline) | 72.4% |
| Random Forest (Optimized) | **85.4%** |

## Academic Context

**Course:** Introduction to Artificial Intelligence (194.025WS)  
**Institution:** TU Wien (Vienna University of Technology)  
**Student:** Emanuel Pitic  

## Technical Stack

- **Language:** Python
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib
- **Model:** Random Forest Classifier
- **Validation:** Stratified K-Fold Cross-Validation
- **Feature Selection:** SelectFromModel with Random Forest

## About Rocket League Trick Shots

Rocket League trick shots are advanced aerial maneuvers that require precise timing and control. This classifier identifies different shot types based on:
- Player movement patterns
- Ball physics (speed, acceleration)
- Spatial relationships (distance to walls, ceiling, ball)
- Input sequences (boost, jump, aerial control)

## License

This project was created for academic purposes as part of coursework at TU Wien.

---

**Author:** Emanuel Pitic  
**GitHub:** [@EmanuelPitic](https://github.com/EmanuelPitic)
