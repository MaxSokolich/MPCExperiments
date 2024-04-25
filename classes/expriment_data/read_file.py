import pandas as pd

# Path to your Excel file
file_path = 'expriment_data/control_action_data.xlsx'

def read_data(file_path):
    # If you are not sure about the sheet names, you can list all sheet names like this
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names  # This will list all sheet names



    # To read a specific sheet by name
    sheet1_df = pd.read_excel(file_path, sheet_name=sheet_names[0])
    sheet2_df = pd.read_excel(file_path, sheet_name=sheet_names[1])
    # Now, let's print the sheet names to see what they are
    # print(sheet_names)
    # 
    # Optionally, print the first few rows of each sheet to verify the content
    # print(sheet1_df.head())
    # print(sheet2_df.head())
    first_non_zero_alpha = sheet1_df[sheet1_df['Alpha'] != 0].index[0]
    ###Combing important rows and save it to df
    start_frame_1 =sheet1_df['Frame'][0]
    start_frame_2 =sheet2_df['Frame'][0]

    end_frame_1 =sheet1_df['Frame'][len(sheet1_df['Frame'])-1]
    end_frame_2 =sheet2_df['Frame'][len(sheet2_df['Frame'])-1]
    end_frame = min(end_frame_1, end_frame_2)
    df = pd.DataFrame([])
    df['Frame'] = sheet1_df['Frame']
    df['Rolling Frequency'] = sheet1_df['Rolling Frequency']
    df['Alpha'] = sheet1_df['Alpha']
    df['Times'] = sheet2_df['Times']
    df['Pos X'] = sheet2_df['Pos X']
    df['Pos Y'] = sheet2_df['Pos Y']
    df['Stuck?'] = sheet2_df['Stuck?']
    # print(df.head())
    # px_idle,py_idle,alpha_idle,time_idle,freq_idle
    return df['Pos X'].tolist(), df['Pos Y'].tolist(), df['Alpha'].tolist(),df['Times'].tolist(), df['Rolling Frequency'].tolist()

read_data(file_path)

