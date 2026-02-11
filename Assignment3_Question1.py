import pandas as pd
import numpy as np

df = pd.read_csv("crime1.csv")

mean = np.mean(df["ViolentCrimesPerPop"])
median = np.median(df["ViolentCrimesPerPop"])
standard_deviation = np.std(df["ViolentCrimesPerPop"])
minimum_value = np.min(df["ViolentCrimesPerPop"])
maximum_value = np.max(df["ViolentCrimesPerPop"])

print(f"Mean: {mean}, Median:  {median}, Standard Deviation: {standard_deviation}, Maximum Value:"
      f" {maximum_value}, Minimum Value: {minimum_value}.")

'''
The mean which is about 0.44 is slightly greater than the median which is 0.39
which means that the distribution is slightly right-skewed.

The maximum value, 1.0 is much further from the median than the minimum
value of 0.02, meaning there are much more higher values.

The mean is more affected by extreme values because it uses all data
values in its calculation, while the median depends only on the middle
value which means only the length matters, not the values of the extreme values.
'''