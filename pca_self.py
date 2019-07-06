# PCA (Principal Component Analysis)

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset importing
data =pd.read_csv('Wine.csv')
X =data.iloc[:,0:13].values
y=data.iloc[:,-1].values

# Spliting into training and test set
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y=train_test_split(X, y, test_size=0.20, random_state=0)

# Feature scaling on data
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
train_X= sc_x.fit_transform(train_X)
test_X=sc_x.transform(test_X)

# to get the analysis varience ration of all features PCA
'''                                       
from sklearn.decomposition import PCA
pca= PCA(n_components=None)
train_X= pca.fit_transform(train_X)
test_X= pca.fit_transform(test_X)
extracted_varience=pca.explained_variance_ratio_  '''

# Applying PCA
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
train_X= pca.fit_transform(train_X)
test_X= pca.transform(test_X)
extracted_varience=pca.explained_variance_ratio_


# Apllying logistic clasification model
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(train_X,train_y)


# Predicting the output of model
y_pred= classifier.predict(test_X)

# Testing the outcome
from sklearn.metrics import confusion_matrix,accuracy_score
cm= confusion_matrix(test_y, y_pred)
score=accuracy_score(test_y, y_pred)*100


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = train_X,  train_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = test_X, test_y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()