import os
import pandas as pd

# specify the directory you're starting from
rootDir = 'files'

# create an empty list to hold dataframes
dfs = []

# loop through your directory
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        if fname.endswith('.csv'):  # make sure the file is CSV
            path = dirName + '/' + fname  # get the file path
            df = pd.read_csv(path, header=None)  # read the csv to a dataframe
            dfs.append(df)  # append the dataframe to the list

# concatenate all dataframes
merged_df = pd.concat(dfs)

# write the dataframe object into csv file
merged_df.to_csv("output.csv", index=None, header=False)

print("Merge operation completed successfully!")


# specify the directory you're starting from
# rootDir = '~/projects/Cuckoo/scripts/pedro'

# # loop through your directory
# fileList = os.listdir("./")
# for fname in fileList:
#     if fname.startswith('slurm'):  # check if the file starts with 'slurm'
#         with open(fname, 'r') as file:  # open the file
#             content = file.read().strip()  # read and strip whitespace
#             print(content)
#             if content != "----------------TRY----------------":
#                 print(f"{fname} has more content than just '----------------TRY----------------'")

