from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import decomposition

#########################################
# Exercise 1
in_dir = "C:/Users/lukas/OneDrive/Uni/Billedanalyse/DTUImageAnalysis-main/DTUImageAnalysis-main/exercises/ex1b-PCA/data/"
txt_name = "irisdata.txt"

iris_data = np.loadtxt(in_dir + txt_name, comments="%")
# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]

n_feat = x.shape[1]
n_obs = x.shape[0]
print(f"Number of features: {n_feat} and number of observations: {n_obs}")

#########################################
# Exercise 2
sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Use ddof = 1 to make an unbiased estimate
var_sep_l = sep_l.var(ddof=1)
var_sep_w = sep_w.var(ddof=1)
var_pet_l = pet_l.var(ddof=1)
var_pet_w = pet_w.var(ddof=1)

#########################################
# Exercise 3
def compute_covariance(x, y):
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)

cov_sepal_length_width = compute_covariance(sep_l, sep_w)
cov_sepal_length_petal = compute_covariance(sep_l, pet_l)

print(f"Covariance between sepal length and width: {cov_sepal_length_width}")
print(f"Covariance between sepal length and petal length: {cov_sepal_length_petal}")

#########################################
# Exercise 4
#plt.figure() # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
#d = pd.DataFrame(x, columns=["Sepal length", "Sepal width", "Petal length", "Petal width"])
#sns.pairplot(d)
#plt.show()

#########################################
# Exercise 5
mn = np.mean(x, axis=0)
data = x - mn

cov_matrix_manual = np.matmul(data.T, data) / (len(data) - 1)
print(cov_matrix_manual)

cov_matrix_np = np.cov(x, rowvar=False)
print(cov_matrix_np)

#########################################
# Exercise 6-7
values, vectors = np.linalg.eig(cov_matrix_manual) # Here cov_matrix_manual is the covariance matrix.

# Compute the proportion of the total variation explained by the first component
total_variation = np.sum(values)
first_component_variation = values[0] / total_variation
print(f"The first component explains {first_component_variation * 100}% of the total variation.")

v_norm = values / values.sum() * 100
plt.plot(v_norm)
plt.xlabel("Principal component")
plt.ylabel("Percent explained variance")
plt.ylim([0, 100])
plt.show()

#########################################
# Exercise 8
pc_proj = vectors.T.dot(data.T)
df_pc_proj = pd.DataFrame(pc_proj.T)

sns.pairplot(df_pc_proj)
plt.show()

#########################################
# Exercise 9
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)
df_data_transform = pd.DataFrame(data_transform)

sns.pairplot(df_data_transform)
plt.show()
