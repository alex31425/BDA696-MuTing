from sklearn import metrics, tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

print("***RandomForest***")
# data source :
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline_RFC = Pipeline(
    [("Normalizer", Normalizer()), ("RandomForest", RandomForestClassifier())]
)
pipeline_RFC.fit(X_train, y_train)
y_pred = pipeline_RFC.predict(X_test)
print(
    " Accuracy - RandomForestClassifier :",
    metrics.accuracy_score(y_test, y_pred),  # flake8: noqa
)  # flake8: noqa

print("***Decision tree***")

pipeline_DT = Pipeline(
    [
        ("Normalizer", Normalizer()),
        ("DecisionTree", tree.DecisionTreeClassifier()),
    ]  # flake8: noqa
)
pipeline_DT.fit(X_train, y_train)
y_predDT = pipeline_DT.predict(X_test)
print(" Accuracy - DecisionTree :", metrics.accuracy_score(y_test, y_predDT))
