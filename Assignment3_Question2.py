import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import title

df = pd.read_csv("crime1.csv")
violent_crimes = df["ViolentCrimesPerPop"]

plt.figure()
plt.hist(violent_crimes)
plt.title("Distribution of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(violent_crimes)
plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("Violent Crimes per Population")
plt.ylabel("Value")
plt.show()

'''
The histogram shows that most Violent Crimes Per Population values are concentrated
toward the lower end, with less observations as values increase.
This means the data is not evenly distributed and has a right-skewed shape.

The box plot shows the median below the center of the box, confirming the skewness.
The upper end extends further than the lower end.
Points outside the box shows that there are outliers,
meaning some areas have unusually high and some unusually low violent crime levels.
'''