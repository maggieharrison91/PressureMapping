import numpy as np
import csv
import pandas 
from PIL import Image

# change date here to get pressure maps for that data
date = "09012025"
path_name = date + "_singleconfig8_pressure_cap_"
paths = []

# creates list of all paths
for i in range(0, 4):
    for j in range(4, 8):
        paths.append(path_name + "CH" + str(i) + "_CH" + str(j) + ".csv")

# reads all csv files by column (channel) and normalizes them
path = "09282025_singleconfig8_pressure_capacitance_CH0_CH4.csv"
column_names = ['timestamp', 'CH0_pF', 'CH1_pF', 'CH2_pF', 'CH3_pF', 'CH4_pF', 'CH5_pF', 'CH6_pF', 'CH7_pF']
df = pandas.DataFrame(pandas.read_csv(path), columns=column_names)

timestamp_data = df['timestamp']
# cutoff = df['timestamp'].str.contains('120')
# print(cutoff)

df_no_time = df.drop(columns=['timestamp'])
max = df_no_time.max().max()
min = df_no_time.min().min()
normalized_df=(df_no_time-min)/(max-min)

normalized_df['timestamp'] = timestamp_data

print(normalized_df)

image = Image.new('P', (4,4), (0, 0, 0))
image.putpixel((0, 0), (0, 0, 255))
resized_image = image.resize((400, 400), Image.Resampling.LANCZOS)
resized_image.show()