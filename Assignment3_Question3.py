import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("kidney_disease.csv")
x = df.drop(columns=["classification"])
y = df["classification"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=67)

'''
We should not train and test a model on the same data because the model
could memorize the training examples instead of learning general
patterns, which would give a misleading performance.

The purpose of the testing set is to evaluate how well the trained model
generalizes to unseen data, providing a more realistic estimate of its
performance on new real-world inputs.
'''