import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("/Users/sagarkumbhar/Documents/heart/heart.csv")

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# trin decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# trained models
model1 = joblib.load('/Users/sagarkumbhar/Documents/heart/trained_models/trained_knn_model.joblib')
model2 = joblib.load('/Users/sagarkumbhar/Documents/heart/trained_models/trained_svm_model.joblib')
model3 = joblib.load('/Users/sagarkumbhar/Documents/heart/trained_models/trained_dtree_model.joblib')
model4 = joblib.load('/Users/sagarkumbhar/Documents/heart/trained_models/trained_rfc_model.joblib')

# Make predictions
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)

# Combine the predictions using a Decision Tree
ensemble_inputs = np.column_stack((y_pred1, y_pred2, y_pred3, y_pred4))
ensemble_model = DecisionTreeClassifier(random_state=42)
ensemble_model.fit(ensemble_inputs, y_test)

# Make predictions using the ensemble model
y_pred = ensemble_model.predict(ensemble_inputs)

# Evaluate the performance of the ensemble model
ensemble_accuracy = accuracy_score(y_test, y_pred)
print("Ensemble accuracy:", ensemble_accuracy)

# Save the trained model
joblib.dump(ensemble_model, 'ens_dtree_model.joblib')

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Plot ROC Curve using Matplotlib
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
plt.plot([0, 1], [0, 1], 'k--', label='No skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()