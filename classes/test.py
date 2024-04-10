import pandas as pd
import numpy as np


df = pd.read_excel("/Users/bizzarohd/Desktop/control_actions.xlsx")

for index, row in df.iterrows():
    # Access the data in each column for the current row
    alpha = row['Alpha']
    freq = row['Rolling Frequency']
    # Process the data as needed
    print(f'Row {index}: {alpha}, {freq}')