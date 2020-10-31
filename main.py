import numpy as np
import pandas as pd

column = [1, 2, 3, 4, "name"]
Iris = pd.read_csv("IRIS.csv", names=column)

# set name to index
AI = Iris.set_index("name")
print(AI)
# unique name
U_name = AI.index.unique()
print(U_name)

# Mean of each feature of each name
N_name = AI.groupby(["name"]).mean()
print(N_name)

# Statistic summary of each feature
print(AI.describe())

# index
index = AI.index.unique()
print(index)
print()

# Transfer DF to NUMPY
arrAI = AI.to_numpy()
arr = Iris.to_numpy()

# Divide data by each name by using For loop
setosa = []
versicolor = []
virginica = []
for i in range(len(arr)):
    if arr[i][4] == U_name[0]:
        for n in arr[i][0:4]:
            setosa.append(n)
    elif arr[i][4] == U_name[1]:
        for n in arr[i][0:4]:
            versicolor.append(n)
    else:
        for n in arr[i][0:4]:
            virginica.append(n)

print("***summary statistics***\n")

print("Mean :")
print(
    f"Mean of setosa is {np.array(setosa).mean()}\
 \nMean of versicolor is {np.array(versicolor).mean()}\
\nMean of virginica is {np.array(virginica).mean()}"
)
print("Min :")
print(
    f"Min of setosa is {np.amin(np.array(setosa))}\
\nMin of versicolor is {np.amin(np.array(versicolor))}\
\nMin of virginica is {np.amin(np.array(virginica))}"
)
print("Max :")
print(
    f"Max of setosa is {np.amax(np.array(setosa))}\
\nMax of versicolor is {np.amax(np.array(versicolor))}\
\nMax of virginica is {np.amax(np.array(virginica))}"
)
print("Quartiles :")
print("Setosa")
print(
    f"Q2 quantile of setosa is {np.quantile(np.array(setosa), 0.5)}\
\nQ1 quantile of setosa is {np.quantile(np.array(setosa), 0.25)}\
\nQ3 quantile of setosa is {np.quantile(np.array(setosa), 0.75)}"
)
print("versicolor")
print(
    f"Q2 quantile of versicolor is {np.quantile(np.array(versicolor), 0.5)}\
\nQ1 quantile of versicolor is {np.quantile(np.array(versicolor), 0.25)}\
\nQ3 quantile of versicolor is {np.quantile(np.array(versicolor), 0.75)}"
)
print("virginica")
print(
    f"Q2 quantile of virginica is {np.quantile(np.array(virginica), 0.5)}\
\nQ1 quantile of virginica is {np.quantile(np.array(virginica), 0.25)}\
\nQ3 quantile of virginica is {np.quantile(np.array(virginica), 0.75)}"
)

print()
print("************************")
