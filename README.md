# BankChurners
A machine learning practice project using Decision Trees to predict credit card customer churn and credit limits. I practiced dealing with data leakage, feature encoding, and used Optuna

In this project, I used a bank customer dataset from Kaggle to solve two different problems:
1. **Classification (Churn):** Trying to predict which customers are likely to close their credit card accounts.
2. **Regression (Credit Limit):** Predicting what a customer's credit limit should be based on their profile.

## 📊 The Dataset
I used the `Credit Card Customers` dataset from Kaggle. It has around 20 features, including customer age, income category, number of transactions, and inactive months. 

## ⚙️ What I Did (My Process)

### 1. Data Cleaning & Fixing "Data Leakage"
* First, I dropped the `CLIENTNUM` column because it's just an ID and doesn't mean anything mathematically.
* I noticed some sneaky columns at the end of the dataset starting with `Naive_Bayes_Classifier`. These were actually predictions left by the original author! I had to drop them so my model wouldn't cheat.
* For the regression part, I also dropped the `Avg_Open_To_Buy` column because it's directly calculated from the credit limit. If I left it in, the model would get a perfect score but wouldn't actually be learning anything.

### 2. Preprocessing (Encoding)
I used Scikit-learn's `ColumnTransformer` to handle the categorical variables:
* **Ordinal Encoding:** For columns that have a natural order (like `Income_Category` and `Education_Level`), I mapped them in the correct hierarchical order (e.g., Less than $40K < $40K - $60K, etc.).
* **One-Hot Encoding:** For nominal columns (like `Gender` and `Marital_Status`), I used one-hot encoding with `drop='first'` to avoid the dummy variable trap.

### 3. The Models
* Built a `DecisionTreeClassifier` to catch the churned customers.
* Built a `DecisionTreeRegressor` to estimate the credit limits.

### 4. Tuning with Optuna! 
Decision trees love to overfit. My base regression model got an $R^2$ score of 0.77, which was okay, but I knew it could be better. I used **Optuna** to run a Bayesian optimization and find the best hyperparameter combo for `max_depth`, `min_samples_split`, and `min_samples_leaf`.
