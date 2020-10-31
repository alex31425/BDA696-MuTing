import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
print(iris.head())
# swarmplot
sns.swarmplot(x="species", y="petal_length", data=iris).set_title("swarmplot")
plt.show()
# stripplot
sns.stripplot(x="species", y="petal_length", data=iris).set_title("stripplot")
plt.show()
# barplot
sns.barplot(x="species", y="petal_length", data=iris).set_title("barplot")
plt.show()
# boxplot
sns.boxplot(x="species", y="petal_length", data=iris).set_title("boxplot")
plt.show()
# violinplot
sns.violinplot(x="species", y="petal_length", data=iris).set_title(
    "violinplot"
)  # flake8: noqa
plt.show()
# scatter
sns.scatterplot(x="species", y="petal_length", data=iris).set_title("scatter")
plt.show()
# pairplot
sns.pairplot(data=iris, hue="species")
plt.show()
