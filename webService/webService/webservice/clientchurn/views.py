from django.shortcuts import render
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import names
#----------------------------------------------------------------------------------------
csv_filename = os.path.join(os.path.dirname(__file__), 'bd_Clients.csv')
maPrediction = []

index = list(range(0,50))
client = []
for i in range(50):
    a = names.get_full_name()
    client.append(a)
bdClients = pd.DataFrame({'client': client}, index = index )

df = pd.read_csv(csv_filename)
df.drop('customerID',axis='columns',inplace=True)
#Suppression des colones vide
df1 = df[df.TotalCharges!=' ']
#Conversion en numerique
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)
#Remplacement des tous les No.. en No
df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)
#Conversion de Yes en 1 Et de No en 0
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)
#Conversion du sexe en 1 et 0
df1['gender'].replace({'Female':1,'Male':0},inplace=True)
#Encodage de la colonne Categorie
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
#Scalaire des colonnes tenure','MonthlyCharges','TotalCharges'
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
#Definition des X et y
X = df2.drop('Churn',axis='columns')
y = df2['Churn']
#Definition des données de test et des données d'entrainements
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
#Construction du modèle (ANN) avec tensorflow/keras
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

def index(request):
    json_records = bdClients.reset_index().to_json(orient='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data}
    return render(request, 'clientchurn/index.html', context)

def clients(request):
    dataframe = df.iloc[:50, :]
    json_records = dataframe.reset_index().to_json(orient='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data}
    return render(request, 'clientchurn/clients.html', context)

def analyses(request):
    df = pd.read_csv(csv_filename)
    df.drop('customerID', axis='columns', inplace=True)
    df1 = df[df.TotalCharges != ' ']
    tenure_churn_no = df1[df1.Churn == 'No'].tenure
    tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure
    plt.plot(tenure_churn_no, label='tenure')
    plt.plot(tenure_churn_yes, label='Number Of Customers')
    plt.legend()
    context = {'plot_div': plt.show()}
    return render(request, 'clientchurn/analyses.html', context)

def predictions(request):
    return render(request, 'clientchurn/predictions.html')

def resultat(request):
    maPrediction.clear()
    custumer_id = request.GET.get('custumer_id')
    print(custumer_id)
    def predict_churn(custumer):
        custumer_to_predict = custumer
        test_pred_custumer = model.predict(custumer)
        print("le resulat de l'algorithm est",test_pred_custumer)
        y_pred = []
        if test_pred_custumer > 0.5:
            y_pred.append(1)
            prediction = "Ce client va partir si on optimise pas les services auquels il est abonné."
        else:
            y_pred.append(0)
            prediction = "Ce client va rester car les services porposés sont pour le moment bien optimisé"
        print("l'arrondi du resultat est:", y_pred)
        return prediction
    prediction = predict_churn(X_test.iloc[[custumer_id]])
    maPrediction.append(prediction)
    print('La prédiction est:', maPrediction)
    print("l'id du client est :", custumer_id)
    context = {
        'prediction': prediction
    }
    return render(request, 'clientchurn/predictions.html', context)