import pandas
from sklearn.preprocessing import *

#Collect CSV file and store it in data structure
def read_file(filepath):
    df = pandas.read_csv(filepath)
    df.dropna(inplace=True)     #Removes all rows with empty cells
    return df

#Preprocessing
#Normalize values, minmax, z-score, robust scaling
def standard_scaling(dataframe, column):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe[[column]])
    return scaled_data

#def normalize

#Processing


