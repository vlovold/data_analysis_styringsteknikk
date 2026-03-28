#Analysis
from scipy import stats
import pandas

#Pearson correlation, Linear disciminant analysis (PCA, LDA)

def pearson_corr(dataframe, column, correlation):
    pear = dataframe[column].corr(dataframe[correlation])
    return pear
#Multi layer perceptron
