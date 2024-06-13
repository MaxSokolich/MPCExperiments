import numpy as np
import pandas as pd
from classes.Learning_module_2d import LearningModule
data = np.load('datasetGP.npy')
df = pd.DataFrame(data)
cycles_gen_data = 3
# Save the DataFrame to an Excel file
# df.to_excel("data.xlsx", index=False)
GP = LearningModule(cycles_gen_data)
GP.read_data_action(data)