import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#https://www.kaggle.com/datasets/uciml/glass

warnings.filterwarnings('ignore')


df = pd.read_csv('glass.csv')


X = df.drop('Type', axis=1)
y = df['Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logistic_params = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial')
logistic_grid = GridSearchCV(logistic_model, logistic_params, cv=5, scoring='accuracy')
logistic_grid.fit(X_train_scaled, y_train)

best_logistic_model = logistic_grid.best_estimator_
logistic_predictions = best_logistic_model.predict(X_test_scaled)

print("Logistic Regression Results:")
print(classification_report(y_test, logistic_predictions, zero_division=1))


logistic_cm = confusion_matrix(y_test, logistic_predictions)
sns.heatmap(logistic_cm, annot=True, fmt="d", cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


tree_params = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}
tree_model = DecisionTreeClassifier()
tree_grid = GridSearchCV(tree_model, tree_params, cv=5, scoring='accuracy')
tree_grid.fit(X_train, y_train)

best_tree_model = tree_grid.best_estimator_
tree_predictions = best_tree_model.predict(X_test)

print("Decision Tree Results:")
print(classification_report(y_test, tree_predictions))


tree_cm = confusion_matrix(y_test, tree_predictions)
sns.heatmap(tree_cm, annot=True, fmt="d", cmap='Greens')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()


feature_importances = pd.DataFrame(best_tree_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances from Decision Tree:")
print(feature_importances)


logistic_coefficients = pd.DataFrame(best_logistic_model.coef_.T, index=X.columns, columns=[f'Class_{i}' for i in range(1, best_logistic_model.coef_.shape[0] + 1)])
print("Logistic Regression Coefficients:")
print(logistic_coefficients)

