#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

import numpy as np

import scipy.stats as s

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.metrics import classification_report


class gaussian_nb:
    
    """Instantiate a Gaussian Naive Bayes Object with the following parameters:
    
    features            : A dataframe consisting of continuous features, exluding labels
    labels              : A series consisting of binary labels
    data_split_ratio    : A tuple consisting of data splitting ratio
    apply_pca           : Boolean value specifying whether to apply PCA or not
    n_components        : Number of Eigen Vectors having Non Zero values to keep
    apply_transform      : Boolean value specifying whether to apply transform or not
    division_number      : slicing for main_data and transform_data
   
    
    """ 
    def __init__(self,features,labels,data_split_ratio, apply_pca,n_components,apply_transform,division_number):
         
        if apply_transform:
            self.transform_data = features.iloc[division_number:]
            self.data = features.iloc[:division_number]
        else:
            self.data = features

        self.binary_labels = np.array(labels).reshape(labels.shape[0], 1)
        self.split_ratio = data_split_ratio
        self.n_principal_components = n_components
        self.unique_labels = list(labels.unique())

        if apply_pca:
            self.X_new, self.mu, self.Q_tilda = self.apply_dim_reduction(self.data, self.n_principal_components)
            self.X_transformed = self.transform(self.data)
        else:
            self.X_new = np.array(self.data)

    def apply_dim_reduction(self, data, n_components):
        X = np.array(data)
        mu = np.mean(X, axis=0).reshape(1, -1)
        X_dash = X - mu
        sigma_hat = (1 / data.shape[0]) * np.matmul(X_dash.T, X_dash)
        sigma_hat_decompose = np.linalg.svd(sigma_hat)
        Q = sigma_hat_decompose[0]
        Q_tilda = Q[:, :n_components]
        X_new = np.matmul(X_dash, Q_tilda)
        return X_new, mu, Q_tilda

    def transform(self, data):
        X_transform = np.array(data)
        X_new_dash = X_transform - self.mu
        X_transformed = np.matmul(X_new_dash, self.Q_tilda)
        return X_transformed

    def data_splitting(self,data):
        new_data = data.copy()
        new_data['label'] = new_data.iloc[:, -1]
        training_data_len = int(self.split_ratio[0] * new_data.shape[0])
        neg_training_data = new_data[new_data['label'] == self.unique_labels[0]].iloc[:training_data_len // 2]
        pos_training_data = new_data[new_data['label'] == self.unique_labels[1]].iloc[:training_data_len // 2]
        training_data = pd.concat([neg_training_data, pos_training_data])
        cv_data_len = int(self.split_ratio[1] * new_data.shape[0])
        neg_remaining_data = new_data[new_data['label'] == self.unique_labels[0]].iloc[training_data_len // 2:]
        pos_remaining_data = new_data[new_data['label'] == self.unique_labels[1]].iloc[training_data_len // 2:]
        remaining_data = pd.concat([neg_remaining_data, pos_remaining_data])
        cv_data = remaining_data.iloc[:cv_data_len]
        testing_data = remaining_data.iloc[cv_data_len:]
        return training_data, cv_data, testing_data

    def train_gaussian_nb(self, data):
        mu_hat_pos = np.array(data[data['label'] == self.unique_labels[1]].iloc[:, :self.n_principal_components].mean())
        sigma_hat_pos = np.array(data[data['label'] == self.unique_labels[1]].iloc[:, :self.n_principal_components].cov())
        mu_hat_neg = np.array(data[data['label'] == self.unique_labels[0]].iloc[:, :self.n_principal_components].mean())
        sigma_hat_neg = np.array(data[data['label'] == self.unique_labels[0]].iloc[:, :self.n_principal_components].cov())
        self.neg_likelihood_params = (mu_hat_neg, sigma_hat_neg)
        self.pos_likelihood_params = (mu_hat_pos, sigma_hat_pos)

    def evaluate(self, data):
        inputs = np.array(data.iloc[:, :self.n_principal_components])
        posterior_pos = s.multivariate_normal.pdf(inputs, self.pos_likelihood_params[0], self.pos_likelihood_params[1])
        posterior_neg = s.multivariate_normal.pdf(inputs, self.neg_likelihood_params[0], self.neg_likelihood_params[1])
        boolean_mask = posterior_pos > posterior_neg
        predicted_category = pd.Series(boolean_mask)
        predicted_category.replace(to_replace=[False, True], value=self.unique_labels, inplace=True)
        predicted_results = np.array(predicted_category)
        actual_results = np.array(data['label'])
        print(classification_report(actual_results, predicted_results, target_names=self.unique_labels))



if __name__ == "__main__":
    print("Going to run a Module as a Script")

# So, now we need to evlaluate the following Probability: 
# 
# \begin{equation}
# P(Diagnosis = M | radius mean = x) = P(radius mean = x | Diagnosis = M) P(texturemean = y| Diagnosis = M) P(Diagnosis = M)
# \end{equation}
# 
# Now, in order to evaluate the likelihood probability P(radiusmean = x | Diagnosis = M) P(texturemean = y | Diagnosis = M) which is given by Multivariate Joint Gaussian Distribution PDF. Now, for this PDF, we need to find out the best estimate of the parameters of Multivariate Joint Normal Distribution because we are assuming that our Malignant tumor training data is being sampled from a Multivariate Joint Normal (Gaussian) Distribution. The two parameters will be namely: mu_hat_m and sigma_hat_m.
# 
# \begin{equation}
# P(radiusmean = x | Diagnosis = M)P(texturemean = y | Diagnosis = M) = \left(\frac{1}{\sqrt{2\pi}\hat{\sigma_\text{rM}}}e^{-\frac{(x-\hat{\mu_\text{rM}})^2}{2\hat{\sigma_\text{rM}^2}}}\right)\left(\frac{1}{\sqrt{2\pi}\hat{\sigma_\text{tM}}}e^{-\frac{(y-\hat{\mu_\text{tM}})^2}{2\hat{\sigma_\text{tM}^2}}}\right)
# \end{equation}

# So, now we need to evlaluate the following Probability: 
# 
# \begin{equation}
# P(Diagnosis = B | radius mean = x) = P(radius mean = x | Diagnosis = B) P(texturemean = y| Diagnosis = B) P(Diagnosis = B)
# \end{equation}
# 
# Now, in order to evaluate the likelihood probability P(radiusmean = x | Diagnosis = B) P(texturemean = y | Diagnosis = B) which is given by Multivariate Joint Gaussian Distribution PDF. Now, for this PDF, we need to find out the best estimate of the parameters of Multivariate Joint Normal Distribution because we are assuming that our Malignant tumor training data is being sampled from a Multivariate Joint Normal (Gaussian) Distribution. The two parameters will be namely: mu_hat_b and sigma_hat_b.
# 
# \begin{equation}
# P(radiusmean = x | Diagnosis = B)P(texturemean = y | Diagnosis = B) = \left(\frac{1}{\sqrt{2\pi}\hat{\sigma_\text{rB}}}e^{-\frac{(x-\hat{\mu_\text{rB}})^2}{2\hat{\sigma_\text{rB}^2}}}\right)\left(\frac{1}{\sqrt{2\pi}\hat{\sigma_\text{tB}}}e^{-\frac{(y-\hat{\mu_\text{tB}})^2}{2\hat{\sigma_\text{tB}^2}}}\right)
# \end{equation}

# In[6]:



# In[ ]:




