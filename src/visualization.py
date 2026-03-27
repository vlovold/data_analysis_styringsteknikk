#Visualization
import pandas
#Create diagrams to show the ouptut of the models
def print_histogram(dataframe):
    dataframe.hist(bins=15)
