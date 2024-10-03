# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithm).

The purpose of the analysis is to predict whether a loan is a high-risk loan (1) or a healthy loan (0) based on borrower financial attributes, using machine learning models. This prediction can help financial institutions better manage their risk by identifying potentially high-risk loans in advance.

The dataset includes various financial attributes, such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The goal is to predict the loan status, where 0 indicates a healthy loan, and 1 represents a high-risk loan.

Key variables in the dataset include:
 - loan_size: The amount of the loan.
 - interest_rate: The interest rate of the loan.
 - borrower_income: The income of the borrower.
 - debt_to_income: Debt-to-income ratio.
 - num_of_accounts: The number of accounts the borrower holds.
 - derogatory_marks: Number of derogatory marks on the borrower's record.
 - total_debt: The total debt the borrower has.
 - Target variable: loan_status (0 = healthy loan, 1 = high-risk loan).

Stages of the machine learning process:
 1. Data Preprocessing: Separated the target variable (loan_status) and the feature variables (financial attributes).
 2. Data Splitting: Split the dataset into training and testing sets using the train_test_split function.
 3. Model Training: Trained a logistic regression model on the training data.
 4. Model Prediction: Used the trained model to predict the loan status on the test set.
 5. Model Evaluation: Evaluated the model performance using a confusion matrix, precision, recall, and F1-score.

 For this analysis, we used a Logistic Regression model, which is commonly used for binary classification problems such as this one (predicting a binary outcome: 0 or 1).

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Accuracy: 99% — The model accurately predicted 99% of the loan statuses in the test set.
    * Precision for 0 (Healthy Loan): 1.00 — The model predicted healthy loans with perfect precision, meaning there were no false positives.
    * Recall for 0 (Healthy Loan): 0.99 — The model correctly identified 99% of the actual healthy loans.
    * F1-Score for 0 (Healthy Loan): 1.00 — This indicates an excellent balance between precision and recall for predicting healthy loans.
    * Precision for 1 (High-Risk Loan): 0.85 — Out of all loans predicted as high-risk, 85% were correct.
    * Recall for 1 (High-Risk Loan): 0.91 — The model captured 91% of the actual high-risk loans.
    * F1-Score for 1 (High-Risk Loan): 0.88 — This suggests the model is reasonably good at predicting high-risk loans but leaves some room for improvement.

## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
    The Logistic Regression model performed well overall, with an accuracy of 99%. It has very high precision and recall for healthy loans (0) and fairly good performance for high-risk loans (1), with an F1-score of 0.88 for the high-risk category. This makes it a reliable model for predicting loan status.
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
    Healthy Loan Prediction (0): The model does an excellent job with predicting healthy loans, with almost perfect precision, recall, and F1-score.
    High-Risk Loan Prediction (1): The model performs reasonably well in predicting high-risk loans, but with slightly lower precision (0.85). This means there are some false positives (loans that are incorrectly classified as high-risk), but it still captures most actual high-risk loans with a recall of 0.91.

Based on the results, the Logistic Regression model is recommended for predicting loan statuses. While it performs exceptionally well for healthy loans, it also shows strong performance in predicting high-risk loans, which is critical for mitigating financial risk. If it's more important to avoid missing high-risk loans, the recall of 0.91 is quite strong for this class. However, if avoiding false positives for high-risk loans is more critical, improvements to precision could be explored, perhaps by tuning the model further.
