import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
disease_predict_df=pd.read_csv('Original_Dataset.csv')
disease_predict_df.fillna(0,inplace=True)
column_values =disease_predict_df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4','Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9','Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14','Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
symps = pd.unique(column_values).tolist()
symp = [i for i in symps if str(i) != "0"]
disease_symptom_df = pd.DataFrame(columns=['Disease'] + symp)
disease_symptom_df['Disease']=disease_predict_df['Disease']
disease_predict_df["symptoms"] = [[] for _ in range(len(disease_predict_df))]
for i in range(len(disease_predict_df)):
    row_values = disease_predict_df.iloc[i].values.tolist()
    if 0 in row_values:
        symptoms_list = row_values[1:row_values.index(0)]
    else:
        symptoms_list = row_values[1:]
    disease_predict_df.at[i, "symptoms"] = symptoms_list
symptoms_series = pd.Series(disease_predict_df["symptoms"] )
disease_symptom_df.iloc[:, 1:] = 0
for i in range(len(disease_symptom_df)):
    symptoms = symptoms_series.iloc[i]
    for symptom in symptoms:
        if symptom in symp:
            disease_symptom_df.at[i, symptom] = 1
X = disease_symptom_df.drop(columns=['Disease'])
y = disease_symptom_df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
symp_model_disease = KNeighborsClassifier()
symp_model_disease.fit(X_train, y_train)
y_pred = symp_model_disease.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# //////////////////////////////////////////////////////////////////////////////
doc_vs_disease = pd.read_csv('Doctor_Versus_Disease.csv', header=None, index_col=None, encoding='ISO-8859-1')
dictionary = doc_vs_disease.set_index(0).to_dict()[1]
def predict_specialist(predicted_disease):
    return dictionary.get(predicted_disease, 'Unknown Specialist')
X = disease_symptom_df.drop(columns=['Disease'])
y = disease_symptom_df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
disease_model_surgery = GaussianNB()
disease_model_surgery.fit(X_train, y_train)
y_pred = disease_model_surgery.predict(X_test)
specialist_recommendations = [predict_specialist(disease) for disease in y_pred]
comparison_df = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True).head(len(y_pred)).values,
    'Predicted': y_pred,
    'Specialist_Recommended': [predict_specialist(disease) for disease in y_pred]
})
accuracy = accuracy_score(y_test, y_pred)
# /////////////////////////////////////////////////////////////////////////////
doctors=pd.read_csv('SurgerySpecialist.csv')
specialist_count=pd.read_csv('medical_specialist_counts.csv')
unique_surgery_types = doctors['SURGERY TYPE'].unique()
def normalize(s):
    return s.strip().lower()
normalized_surgery_set = set(normalize(s) for s in unique_surgery_types)
normalized_dict_set = set(normalize(s) for s in dictionary.values())
matching_specialists = normalized_surgery_set.intersection(normalized_dict_set)
Unique_Disease=doctors['Medical Intervention'].unique()
def normalize(s):
    return s.strip().lower()
normalized_surgery_set = set(normalize(s) for s in Unique_Disease)
normalized_dict_set = set(normalize(s) for s in dictionary.keys())
matching_specialists = normalized_surgery_set.intersection(normalized_dict_set)
specific_values = ['Dermatologist', 'Gynecologist', 'Gastroenterologist', 'Cardiologist']
filtered_df = doctors[doctors['SURGERY TYPE'].isin(specific_values)]
doctors=filtered_df
final_df=doctors
final_df['Medical Intervention'] = final_df['Medical Intervention'].fillna('')
final_df=final_df.dropna(axis=0)
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(final_df['SURGERY TYPE'])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(final_df.index, index=final_df['SURGERY TYPE']).drop_duplicates()
def model1(surgery, sig=sig):
    idx = indices[surgery]
    return list(dict.fromkeys(final_df[final_df['SURGERY TYPE']==surgery]['Name']))[:10]
# le_surgery = LabelEncoder()
# final_df['SURGERY TYPE Encoded'] = le_surgery.fit_transform(final_df['SURGERY TYPE'])
# final_df['QUALIFICATIONS Encoded'] = final_df['QUALIFICATIONS'].astype('category').cat.codes
# features = final_df[['SURGERY TYPE Encoded', 'QUALIFICATIONS Encoded']]
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
# knn = NearestNeighbors(n_neighbors=10, algorithm='auto')
# knn.fit(features_scaled)
# def model2(surgery_name):
#     if surgery_name not in final_df['SURGERY TYPE'].values:
#         return "surgery not found."
#     idx = final_df[final_df['SURGERY TYPE'] == surgery_name].index[0]
#     surgery_features = features_scaled[idx].reshape(1, -1)
#     distances, indices = knn.kneighbors(surgery_features)
#     similar_indices = indices[0][1:]
#     return list(dict.fromkeys(final_df['Name'].iloc[similar_indices]))[:10]
referene_df = pd.DataFrame(columns=symp)
listings = []
for i in range(len(symp)):
    listings.append(0)
referene_df.loc[0] = listings
print(referene_df.head())
input_text = input()
input_list = input_text.split(",")
for i in input_list:
    referene_df.at[0,i] = 1
input_predicted = disease_model_surgery.predict(referene_df)
print(input_predicted)
Surgery = ""
for i in input_predicted :
    Surgery = dictionary.get(i)
    print(dictionary.get(i))
print(model1(Surgery))
# print(model2(intervention.iloc[0]))