import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

in_dir = "C:/Users/Christian/Documents/DTU/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1b-PCA/data/"
txt_name = "irisdata.txt"

# Start by reading the data and create a data matrix `x`:
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

# Then check the data dimensions by writing:
n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")

# To explore the data, we can create vectors of the individual feature:
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Compute the variance of each feature
# Use ddof = 1 to make an unbiased estimate
var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1)

# Compute covariance between the features
def compute_covariance(x, y):
    """Compute the covariance between two arrays."""
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)

cov_sepal_length_width = compute_covariance(sep_l, sep_w)
cov_sepal_length_petal_length = compute_covariance(pet_l, pet_w)

#print("Covariance between sepal length and sepal width: ", cov_sepal_length_width)
#print("Covariance between sepal length and petal length: ", cov_sepal_length_petal_length)

# Pair plot
#plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
d = pd.DataFrame(x, columns=["Sepal length", "Sepal width",
							 "Petal length", "Petal width"])
#sns.pairplot(d)
#plt.show()

# compute the covariance matrix using
mn = np.mean(x, axis=0)
data = x - mn

cov_matrix_manual = np.dot(data.T, data) / (len(data) - 1)
#print(cov_matrix_manual)

cov_matrix = np.cov(x, rowvar=False, ddof=1)
#print(cov_matrix)

# Compute the eigenvalues and eigenvectors of the covariance matrix
values, vectors = np.linalg.eig(cov_matrix)

v_norm = values / values.sum() * 100

# Print the percentage of the total variance explained by each principal component
print("The first principal component explains {:.2f}% of the total variance.".format(v_norm[0])) # For the first principal component

plt.plot(v_norm)
plt.xlabel("Principal component")
plt.ylabel("Percent explained variance")
plt.ylim([0, 100])

#plt.show()

# The data can be projected onto the PCA space by using the dot-product
pc_proj = vectors.T.dot(data.T)
pc_proj_d = pd.DataFrame(pc_proj.T)
sns.pairplot(pc_proj_d)
plt.show()

# PCA using scikit-learn
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)
a = pd.DataFrame(data_transform)
sns.pairplot(a)
plt.show()




