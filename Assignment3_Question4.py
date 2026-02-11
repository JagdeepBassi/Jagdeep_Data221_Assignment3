import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("kidney_disease.csv")
x = df.drop(columns=["classification"])
y = df["classification"]

x = x.select_dtypes(include="number")
x = x.fillna(x.mean())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=67)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_prediction = knn.predict(x_test)

confusion_matrix = confusion_matrix(y_test, y_prediction)
accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction, pos_label="ckd")
recall = recall_score(y_test, y_prediction, pos_label="ckd")
f1 = f1_score(y_test, y_prediction, pos_label="ckd")

print("Confusion Matrix:")
print(confusion_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

'''
A True Positive means the model correctly predicts that a patient has
kidney disease, while a True Negative means it correctly predicts that
a patient does not have kidney disease.

A False Positive is when the model predicts kidney disease for a
healthy patient, and a False Negative is when the model fails to
detect kidney disease for a patient with the disease.

Accuracy alone may not be enough because it does not distinguish between
different types of errors, which is important when the classes are
imbalanced or when some errors are more serious than others.

If missing a kidney disease case is very serious, recall is the most
important metric because it measures how well the model identifies
actual positive cases and minimizes false negatives.
'''