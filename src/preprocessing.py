import pandas

#Collect CSV file and store it in data structure
def read_file(filepath):
    df = pandas.read_csv(filepath)
    df.dropna(inplace=True)     #Removes all rows with empty cells
    return df

#Preprocessing
#Find nan values and deal with them

#Processing
#Normalize values, minmax, z-score, robust scaling

