import pandas as pd

icd = pd.read_csv('data/initial_datasets/ICD10codes.csv')

print(icd.head())

# dropping unwanted column and naming the other columns
icd = icd.drop(icd.columns[3], axis = 1)

icd.columns = ["Group", "Subgroup", "Code", "Desc", "Category"]

print(icd.head())

print("Shape: ", icd.shape)

# Dropping initial Duplicates
icd.drop_duplicates()

print("Shape After dropping Duplicates: ", icd.shape)

# correcting wrong entries
icd.iloc[:, 1] = icd.iloc[:, 1].fillna('')

for row in icd.values:
    rem=''
    if len(row[0])>3:
        rem = row[0][3:]

    row[0] = row[0][:3]

    suf = str(row[1])
    row[1] = rem+suf

print(icd.head())

print("Shape: ", icd.shape)

# Dropping initial Duplicates
icd.drop_duplicates()

print("Shape After dropping Duplicates: ", icd.shape)

icd.to_csv('data/final_datasets/final_ICD.csv', index=False)