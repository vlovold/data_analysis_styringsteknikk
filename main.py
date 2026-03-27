import scipy
import matplotlib.pyplot as mpl
import numpy
import sklearn
from src.preprocessing import read_file, standard_scaling
from src.visualization import print_histogram
#Initializes data
df = read_file('equipment_anomaly_data.csv')

#prints histogram and 5 first rows
ax = print_histogram(df)
print(df.head())
#mpl.show()


#Normalizes columns
scaled_temp = standard_scaling(df, 'temperature')
scaled_hum = standard_scaling(df, 'humidity')
scaled_vib = standard_scaling(df, 'vibration')
scaled_pres = standard_scaling(df, 'pressure')

