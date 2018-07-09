import pandas as pd
'''
  This file reads in the diabetes hospital data file 'diabetic_data.csv' 
  into a pandas dataframe, cleans the data, and ouptuts a new (smaller) 
  csv file with cleaned data called 'cleaned_diabetic_data.csv'
'''

# Read csv file with raw data and store in 'df'.
df = pd.read_csv('diabetic_data.csv')

# Remove duplicate patient visits.  Only include a patient's first visit.
df.drop_duplicates(subset='patient_nbr',keep='first', inplace=True)

# List of features
features = ['race','gender','age','discharge_disposition_id','admission_source_id','time_in_hospital','medical_specialty','diag_1','A1Cresult','change','readmitted']

# Delete columns that do not correspond to the wanted features.
df = df[features]

# Remove patients who expired or were discharged to a hospice.
unwanted_discharge_codes = [11,13,14,19,20,21]
df = df[ ~df['discharge_disposition_id'].isin(unwanted_discharge_codes) ]

# Convert race column to a numeric type. 0 if caucasian, 1 if african
# american, 3 if other, and 4 if missing
df.loc[ df['race']=='Caucasian', 'race' ] = 0
df.loc[ df['race']=='AfricanAmerican', 'race' ] = 1
df.loc[ df['race']=='Hispanic', 'race' ] = 3
df.loc[ df['race']=='Asian', 'race' ] = 3
df.loc[ df['race']=='Other', 'race' ] = 3
df.loc[ df['race']=='?', 'race' ] = 4

# Convert gender column to a numeric type.  0 if male, 1 if female, and 2
# if unknown.
df.loc[ df['gender']=='Male', 'gender' ] = 0
df.loc[ df['gender']=='Female', 'gender' ] = 1
df.loc[ df['gender']=='Unknown/Invalid', 'gender' ] = 2

# Convert age column to a numeric type.
# 0 is younger than 30
# 1 is between 30 and 60
# 2 is older than 60
df.loc[ df['age']=='[0-10)', 'age'] = 0
df.loc[ df['age']=='[10-20)', 'age'] = 0
df.loc[ df['age']=='[20-30)', 'age'] = 0
df.loc[ df['age']=='[30-40)', 'age'] = 1
df.loc[ df['age']=='[40-50)', 'age'] = 1
df.loc[ df['age']=='[50-60)', 'age'] = 1
df.loc[ df['age']=='[60-70)', 'age'] = 2
df.loc[ df['age']=='[70-80)', 'age'] = 2
df.loc[ df['age']=='[80-90)', 'age'] = 2
df.loc[ df['age']=='[90-100)', 'age'] = 2

# Convert medical specialty to a numeric type.
# cardiology = 0, Family/GeneralPractice = 1
# internalmedicine = 2, surgery = 3
# missing = 4, other = 5
surg = []
for ms in df['medical_specialty']:
    if (ms[:4] =='Surg'):
        surg.append(ms)
surg = list(set(surg))

df.loc[ df['medical_specialty']=='Cardiology', 'medical_specialty'] = 0
df.loc[ df['medical_specialty']=='Family/GeneralPractice', 'medical_specialty'] = 1
df.loc[ df['medical_specialty']=='InternalMedicine', 'medical_specialty'] = 2
for i in range( len(surg)):
    df.loc[ df['medical_specialty']== surg[i], 'medical_specialty'] = 3
df.loc[ df['medical_specialty']=='?', 'medical_specialty'] = 4

other = []
for o in df['medical_specialty']:
    if isinstance(o,str):
        other.append(o)
other = list(set(other))
for i in range( len(other)):
    df.loc[ df['medical_specialty']== other[i], 'medical_specialty'] = 5

# Convert HbA1c to a numeric type. No test = 0, 
# test with normal results = 1
# high, with no change in medication =2
# high, with change = 3
df.loc[ df['A1Cresult']=='None', 'A1Cresult' ] = 0
df.loc[ df['A1Cresult']=='Norm', 'A1Cresult' ] = 1
df.loc[ (df['A1Cresult']=='>7')&(df['change']=='No'), 'A1Cresult' ] = 2
df.loc[ (df['A1Cresult']=='>8')&(df['change']=='No'), 'A1Cresult' ] = 2
df.loc[ (df['A1Cresult']=='>7')&(df['change']=='Ch'), 'A1Cresult' ] = 3
df.loc[ (df['A1Cresult']=='>8')&(df['change']=='Ch'), 'A1Cresult' ] = 3

# Convert primary diagnosis to numeric type.
# 0 = circulatory
# 1 = diabetes
# 2 = respiratory
# 3 = digestive
# 4 = injury and poisoning
# 5 = musculoskeletal
# 6 = genitourinary
# 7 = neoplasms
# 8 = other
df['diag_1'] = pd.to_numeric( df['diag_1'], errors='coerce' )

df.loc[ (df['diag_1']>=0) & (df['diag_1']<=139), 'diag_1'] = 8
df.loc[ (df['diag_1']>=240) & (df['diag_1']<=249), 'diag_1'] = 8
df.loc[ (df['diag_1']>=251) & (df['diag_1']<=389), 'diag_1'] = 8
df.loc[ (df['diag_1']>=630) & (df['diag_1']<=709), 'diag_1'] = 8
df.loc[ (df['diag_1']>=740) & (df['diag_1']<=784), 'diag_1'] = 8
df.loc[ (df['diag_1']>=789) & (df['diag_1']<=799), 'diag_1'] = 8
df.loc[ (df['diag_1']>=1000), 'diag_1'] = 8
df.loc[ (pd.isna(df['diag_1'])), 'diag_1'] = 8


df.loc[ (df['diag_1']>=390) & (df['diag_1']<=459), 'diag_1'] = 0
df.loc[ df['diag_1'] == 785.0, 'diag_1'] = 0

df.loc[ (df['diag_1']>=250.0) & (df['diag_1']<251), 'diag_1'] = 1

df.loc[ (df['diag_1']>=460) & (df['diag_1']<=519), 'diag_1'] = 2
df.loc[ df['diag_1'] == 786.0, 'diag_1'] = 2

df.loc[ (df['diag_1']>=520) & (df['diag_1']<=579), 'diag_1'] = 3
df.loc[ df['diag_1'] == 787.0, 'diag_1'] = 3

df.loc[ (df['diag_1']>=800) & (df['diag_1']<=999), 'diag_1'] = 4

df.loc[ (df['diag_1']>=710) & (df['diag_1']<=739), 'diag_1'] = 5

df.loc[ (df['diag_1']>=580) & (df['diag_1']<=629), 'diag_1'] = 6
df.loc[ df['diag_1'] == 788.0, 'diag_1'] = 6

df.loc[ (df['diag_1']>=140) & (df['diag_1']<=239), 'diag_1'] = 7

df['diag_1'] = df['diag_1'].astype('int64')

# Convert readmitted column to a numeric type. 0 if not readmitted, 1 if
# readmitted.
df.loc[ df['readmitted']!='NO', 'readmitted'] = 1
df.loc[ df['readmitted']=='NO', 'readmitted'] = 0

# Remove 'change' columns (not needed in data analysis, because this
# information is included in the 'A1Cresult' columns)
df.drop( columns=['change'], inplace=True)

# Output dataframe into a new csv file.
df.to_csv('diabetic_data_cleaned.csv',index = False)
