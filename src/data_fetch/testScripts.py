import pandas as pd

file_path = 'data/exofop_tess_tois.csv'

toi_list = pd.read_csv(file_path, skiprows=1)



toi_list = toi_list[
    (toi_list['TESS Disposition'].isin(['EB', 'V'])) | 
    (toi_list['TFOPWG Disposition'].isin(['FP', 'FA']))
]

tic_ids = toi_list['TIC ID'].tolist()
#tic_ids = tic_ids[396:]
print(tic_ids.index(399956873))
#print(tic_ids[0])

'''

distinct_entries1 = df["TESS Disposition"].unique()
distinct_entries2 = df["TFOPWG Disposition"].unique()

# Print the distinct entries
print(distinct_entries1)
print(distinct_entries2)
'''
