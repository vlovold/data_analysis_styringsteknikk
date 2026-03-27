import scipy
import matplotlib.pyplot as mpl
import numpy
import sklearn
from src.preprocessing import read_file
from src.visualization import print_histogram

df = read_file('equipment_anomaly_data.csv')
ax = print_histogram(df)
print(df.head())

mpl.show()

