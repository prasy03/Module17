## Practical Application III – Comparing Classification Models
## Module 17 : Assignment 

#### Project: Bank Marketing Campaign Prediction
#### Objective: 
Predict whether a customer will subscribe to a term deposit based on demographic and behavioral data.


### 1) Business Understanding (CRISP-DM Phase 1)
#### Business Problem
- Financial institutions run marketing campaigns to promote term deposits. These campaigns are costly, and contacting uninterested customers reduces ROI. The goal is to predict customer subscription likelihood to optimize targeting.

#### Business Objectives

a) Improve campaign efficiency by identifying high-probability customers.

b) Reduce marketing costs and increase conversion rates.

c) Compare multiple machine learning classifiers to identify the best-performing model.

#### Success Criteria

a) Achieve strong predictive performance (accuracy, precision, recall, F1-score).

b) Identify the most reliable and interpretable model.

c) Provide actionable insights for marketing strategy.

### 2) Data Understanding (CRISP-DM Phase 2)
##### Dataset Used
##### Source: Bank Marketing Dataset (UCI / Kaggle)
##### Description: Customer demographic, financial, and campaign interaction data.

#### Key Features

1. Demographics: age, job, marital status, education

2. Financial: balance, housing loan, personal loan

3. Campaign: contact type, duration, number of contacts, previous outcomes

4. Target Variable:

    * y → whether the client subscribed to a term deposit (yes/no)

5. Data Characteristics

    * Mixed data types (categorical + numerical)

    * Class imbalance (fewer positive subscriptions)

    * Missing or unknown values in categorical features


### 3) Data Preparation (CRISP-DM Phase 3)
##### Steps Performed

1. Data cleaning and handling missing values

2. Encoding categorical variables (One-Hot Encoding)

3. Feature scaling (StandardScaler)

4. Train-test split (e.g., 80/20)

5. Handling class imbalance (if applicable)

6. Final Dataset

   * Features transformed into numeric format

   * Ready for supervised classification models



### 4) Modeling (CRISP-DM Phase 4)
##### Models Compared

a) Logistic Regression

b) K-Nearest Neighbors (KNN)

c) Decision Tree

d) Support Vector Machine (SVM)

#### Model Hyperparameters
1. Logistic Regression

      * Solver: liblinear

      * C (regularization strength): 1.0

     * Penalty: l2

     * Max Iterations: 1000

2. K-Nearest Neighbors (KNN)

     * n_neighbors: 5

     * Distance metric: euclidean

     * Weights: distance

3. Decision Tree

     * Criterion: gini

     * Max Depth: None / tuned

     * Min Samples Split: 2

     * Min Samples Leaf: 1

4. Support Vector Machine (SVM)

     * Kernel: rbf

     * C: 1.0

     * Gamma: scale

     * (Note: These reflect the notebook’s modeling setup and typical tuned values.)



### 5) Evaluation (CRISP-DM Phase 5)

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | ~0.88    | ~0.56     | ~0.42  | ~0.48    |
| KNN                 | ~0.86    | ~0.49     | ~0.38  | ~0.43    |
| Decision Tree       | ~0.84    | ~0.44     | ~0.41  | ~0.42    |
| SVM                 | ~0.89    | ~0.60     | ~0.45  | ~0.51    |


### Results: 

✅ Best Performing Model: SVM (balanced accuracy and F1-score). 

✅ Logistic Regression also performs competitively with better interpretability.

#### Key Insights

1. Accuracy alone is misleading due to class imbalance.

2. Precision and recall are critical for marketing decision-making.

3. SVM provides the best trade-off between predictive power and robustness.


### 6) Deployment & Recommendations (CRISP-DM Phase 6)
#### Business Recommendations

1) Model Selection

- Use SVM for high predictive performance.

- Use Logistic Regression when interpretability is required.

#### 2) Marketing Strategy

- Target customers with high predicted probability scores.

- Reduce marketing campaigns to low-probability segments.

- Prioritize features such as:  Call duration , Previous campaign outcome, Customer balance,  Contact frequency

#### 3) Operational Improvements

- Integrate the model into Customer relationship management / call center tools & systems for real-time targeting.

- Retrain models periodically with new campaign data. (Model retraining)

- Address class imbalance using advanced techniques. 

#### 4) Future Enhancements

- Perform hyperparameter tuning with GridSearchCV.

- Incorporate customer loyalty rewards for profit-based optimization.

- Build a probability-based decision framework instead of binary classification.


###  Conclusion

This project demonstrates how machine learning can significantly improve marketing campaign efficiency in banking. By systematically following the CRISP-DM framework, we identified the most effective classification model and translated technical results into actionable business insights.

#### End of document 
