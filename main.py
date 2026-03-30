import scipy
import matplotlib.pyplot as mpl
import numpy
import sklearn
from src.preprocessing import read_file, standard_scaling
from src.visualization import print_histogram, plot_correlation_heatmap
from src.models import pearson_corr, linear_discriminant

#Initializes data
df = read_file('equipment_anomaly_data.csv')

#prints histogram and 5 first rows
ax = print_histogram(df)
print(df.head())
#mpl.show()


#Normalizes columns and create new dataframe
scaled_temp = standard_scaling(df, 'temperature')
scaled_hum = standard_scaling(df, 'humidity')
scaled_vib = standard_scaling(df, 'vibration')
scaled_pres = standard_scaling(df, 'pressure')

df_normalized = df.copy()
df_normalized['temperature'] = scaled_temp
df_normalized['humidity'] = scaled_hum
df_normalized['vibration'] = scaled_vib
df_normalized['pressure'] = scaled_pres

#Analysis

#print(pearson_corr(df, 'pressure', 'faulty'))

#Plot correlation heatmap
#plot_correlation_heatmap(df_normalized)

print(linear_discriminant(df_normalized))
