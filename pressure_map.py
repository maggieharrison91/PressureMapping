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
        paths.append(path_name + "CH" + str(i) + "_CH" + str(j) + "_filtered_truncated.csv")

# reads all csv files by column (channel) and normalizes them
path = "10142025_singleconfig8_pressure_cap_CH0_CH4_filtered_truncated.csv"
column_names = ['timestamp', 'CH0_pF_filtered', 'CH1_pF_filtered', 'CH2_pF_filtered', 'CH3_pF_filtered', 'CH4_pF_filtered', 'CH5_pF_filtered', 'CH6_pF_filtered', 'CH7_pF_filtered']
df = pandas.DataFrame(pandas.read_csv(path), columns=column_names)

timestamp_data = df['timestamp']
# cutoff = df['timestamp'].str.contains('120')
# print(cutoff)

df_no_time = df.drop(columns=['timestamp'])
max = df_no_time.max().max()
min = df_no_time.min().min()
normalized_df = (df_no_time - min) / (max - min)

normalized_df['timestamp'] = timestamp_data

print(normalized_df)

print(normalized_df[round(normalized_df['timestamp']) == 120])
print(df[round(df['timestamp']) == 120])

# for i in range (0, 4):
#     for j in range(4, 8):
        

# makes the image for pressure mapping
image = Image.new('P', (4,4), (0, 0, 0))
image.putpixel((0, 0), (0, 0, 255))
resized_image = image.resize((400, 400), Image.Resampling.LANCZOS)
resized_image.show()


## TODO: Normalization (same as what they do for their data), Generate pressure map