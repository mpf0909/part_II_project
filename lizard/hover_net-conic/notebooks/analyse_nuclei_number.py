import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

### logistic regression just considering total number of nuclei ###

normal_df = pd.read_csv('../toy_data_20x_inference/normal/225363019/valid_pred_cell.csv')
coeliac_df = pd.read_csv('../toy_data_20x_inference/coeliac/235364597/valid_pred_cell.csv')

# Sum all nuclei counts per image
normal_df['total_nuclei'] = normal_df.sum(axis=1)
coeliac_df['total_nuclei'] = coeliac_df.sum(axis=1)

# Add labels: 0 for normal, 1 for coeliac
normal_df['label'] = 0
coeliac_df['label'] = 1

# Combine datasets
combined_df = pd.concat([normal_df, coeliac_df], ignore_index=True)

# Prepare features and labels
X = combined_df[['total_nuclei']]
y = combined_df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Output model coefficients
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_[0])