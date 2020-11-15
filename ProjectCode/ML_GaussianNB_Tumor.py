from sklearn.datasets import load_breast_cancer  # Loading breast cancer dataset
from sklearn.model_selection import train_test_split    # Useful to split the dataset randomly into train and test
from sklearn.naive_bayes import GaussianNB              # Training model
from sklearn.metrics import accuracy_score              # Function to get model accuracy

# Load dataset
data = load_breast_cancer()

# Organize our data
label_names = data['target_names']      # malignant (0) benign (1)
labels = data['target']                 # yp = 0 or 1s according to each tumor metric
feature_names = data['feature_names']   # 30 features
features = data['data']                 # 30 rows

# Look at our data
print(label_names)
print(feature_names)
print(features[0])    # mean radius

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)  # random state is the seed for randomness

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
preds = gnb.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))
