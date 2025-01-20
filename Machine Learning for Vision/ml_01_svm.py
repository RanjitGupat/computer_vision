# Scikit-learn has emerged as a powerful and user-friendly Python library. 
# Its simplicity and versatility make it a better choice for both beginners 
# and seasoned data scientists to build and implement machine learning models. 
# In this article, we will explore about Sklearn.

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict the value of the digit on the test set
predicted = classifier.predict(X_test)

# Print the classification report
print(metrics.classification_report(y_test, predicted))