#Analysis
from scipy import stats
import pandas

#Pearson correlation, Linear disciminant analysis (PCA, LDA)

def pearson_corr(dataframe, column, correlation):
    matrix = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(stats.pearsonr(dataframe.iloc[i], dataframe.iloc[j]))
    r, p = stats.pearsonr(dataframe[column], dataframe[correlation])
    return r
#Multi layer perceptron
