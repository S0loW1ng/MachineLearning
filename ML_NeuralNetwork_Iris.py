import pandas as pd

# Dataset Location

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to dataset

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Reading to pandas dataframe

irisdata = pd.read_csv(url, names=names)

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])

# irisdata.head()
# Converting labels to numerical labels using Scikit-Learn's LabelEncoder class

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
# y.Class.unique() to check type of values in
# y.head() to show how y looks

# Splitting data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature scaling (standardizing data so there is no extra weight due to big data values)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training and predictions

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# 3 layers 10 nodes each
# max_iter is one epoch which is a combination
# of one cycle of feed-forward and back propagation phase
# by default, relu activation func is used with 'adam' cost optimizer, can change
# by using activation and solver parameters

mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
