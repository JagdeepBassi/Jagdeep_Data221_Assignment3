import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("kidney_disease.csv")
x = df.drop(columns=["classification"])
y = df["classification"]

x = x.select_dtypes(include="number")
x = x.fillna(x.mean())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=67)

k_values = [1, 3, 5, 7, 9]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_prediction = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    results.append((k, accuracy))

results_df = pd.DataFrame(results, columns=["k", "Test Accuracy"])
print(results_df)

best_k = results_df.loc[results_df["Test Accuracy"].idxmax(), "k"]
print(f"Highest test accuracy is achieved at k = {best_k}")

'''
Increasing k makes the model consider more neighbors when predicting,
which smooths the decision boundary and reduces variance.

Very small k may cause overfitting because the model
perfectly memorizes training points, capturing noise as well as signal.

Very large k can cause underfitting because the model averages over
too many neighbors, ignoring local patterns and making predictions
too general.
'''