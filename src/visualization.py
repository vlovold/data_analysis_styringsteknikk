#Visualization
import pandas
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import pyplot as plt


#Create diagrams to show the ouptut of the models
def print_histogram(dataframe):
    dataframe.hist(bins=15)


#Created a plot correlation heat map, uses original numbers, not scaled
def plot_correlation_heatmap(dataframe):
    matrix = dataframe.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap = "coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Viktor & Elisabeths cool Correlation Matrix")
    plt.show()

