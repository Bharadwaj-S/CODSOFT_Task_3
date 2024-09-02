#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, classification_report

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Dropping non-relevant columns
irrelevant_columns = ['RowNumber', 'CustomerId', 'Surname']
X = df.drop(columns=['Exited'] + irrelevant_columns)
y = df['Exited']

# Identify categorical and numerical features
categorical_features = ['Geography', 'Gender']
numeric_features = X.columns.difference(categorical_features)

# Preprocessing pipeline for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline to training and test data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")

# Decision Tree Model
dtree = DecisionTreeClassifier(max_depth=5)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f"Decision Tree Accuracy: {accuracy_dtree:.4f}")

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Gradient Boosting Model
gbc = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Gradient Boosting Accuracy: {accuracy_gbc:.4f}")

# Evaluate on the test set with the best-performing model
best_model = rf  # Replace with gbc if Gradient Boosting performs better

y_test_pred = best_model.predict(X_test)
print("\nTest Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ROC Curve and AUC
y_test_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Feature Importance Plot
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
    features = np.hstack([numeric_features, preprocessor.named_transformers_['cat'].get_feature_names_out()])
    sorted_idx = np.argsort(feature_importances)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance in the Best Model")
    plt.show()

# Learning Curve
train_sizes = np.linspace(0.1, 0.9, 10)  # Avoiding 1.0
train_scores = []
test_scores = []

for train_size in train_sizes:
    X_train_part, _, y_train_part, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    best_model.fit(X_train_part, y_train_part)
    train_scores.append(accuracy_score(y_train_part, best_model.predict(X_train_part)))
    test_scores.append(accuracy_score(y_test, best_model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training accuracy')
plt.plot(train_sizes, test_scores, 'o-', color='green', label='Test accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()
