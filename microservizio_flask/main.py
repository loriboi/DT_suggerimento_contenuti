from flask import Flask, request, jsonify, Response


#dipendenze

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import json
import joblib

import numpy as np
app = Flask(__name__)


main_folder = 'path/della/cartella/modelli/map'

@app.route("/predict", methods=["POST"])
def predict_image():

    req = request.json

    def map_record1(df,categorie_numeri,nazionalita_numeri,occupazione_numeri):
        df["categoria_preferita1"] = df["categoria_preferita1"].map(categorie_numeri)
        df["categoria_preferita2"] = df["categoria_preferita2"].map(categorie_numeri)
        df["categoria_preferita3"] = df["categoria_preferita3"].map(categorie_numeri)
        df["nazionalita_utente"] = df["nazionalita_utente"].map(nazionalita_numeri)
        df["occupazione_utente"] = df["occupazione_utente"].map(occupazione_numeri)
        return df

    def map_record2(df,categorie_numeri,nazionalita_numeri,occupazione_numeri,contenuto_numeri,attore_numeri,regista_numeri):
        df["categoria_preferita1"] = df["categoria_preferita1"].map(categorie_numeri)
        df["categoria_preferita2"] = df["categoria_preferita2"].map(categorie_numeri)
        df["categoria_preferita3"] = df["categoria_preferita3"].map(categorie_numeri)
        df["nazionalita_utente"] = df["nazionalita_utente"].map(nazionalita_numeri)
        df["occupazione_utente"] = df["occupazione_utente"].map(occupazione_numeri)
        return df

    def verifica_utente(dictionary_user):
        attributi_json1 = {'eta_utente', 'nazionalita_utente', 'occupazione_utente', 'categoria_preferita1', 'categoria_preferita2', 'categoria_preferita3'}
        attributi_json2 = {'eta_utente', 'nazionalita_utente', 'occupazione_utente', 'categoria_preferita1', 'categoria_preferita2', 'categoria_preferita3', 'durata_media_sessioni', 'durata_contenuto_preferito'}
        if all(attributo in dictionary_user for attributo in attributi_json2):
            return 2
        elif all(attributo in dictionary_user for attributo in attributi_json1):
            return 1
        else:
            return -1
    
    def classify_utente(dictionary_user,kmeans1,kmeans2,map_cat,map_naz,map_occ,map_cont,map_act,map_reg):
        type_user = verifica_utente(dictionary_user)
        if type_user == 1:
            new_record1 = pd.DataFrame(dictionary_user)
            new_record1 = map_record1(new_record1,map_cat,map_naz,map_occ)
            new_group1 = kmeans1.predict(new_record1)
            new_group1 = new_group1[0]
            cluster = new_group1
            return (cluster,type_user)
        elif type_user == 2:
            new_record2 = pd.DataFrame(dictionary_user)
            new_record2 = map_record2(new_record2,map_cat,map_naz,map_occ,map_cont,map_act,map_reg)
            new_group2 = kmeans2.predict(new_record2)
            new_group2 = new_group2[0]
            cluster = new_group2
            return (cluster,type_user)
        elif verifica_utente(dictionary_user) == -1:
            return (-1,-1)  
  
    def favourite_contents(model,df,cluster,mappa):
        df = df[df['Cluster'] == cluster]
        preferred_contents = df.groupby('Cluster')['contenuto_preferito'].apply(lambda x: x.value_counts().nlargest(2).index.tolist())
        mappa_invertita = {numero: nome for nome, numero in mappa.items()}
        preferred_names = preferred_contents.apply(lambda x: [mappa_invertita[numero] for numero in x])
        return preferred_names

    def get_suggestions(kmeans1,kmeans2,typeofuser,cluster,df1,df2,mappa):
        if typeofuser == 1:
            suggestions = favourite_contents(kmeans1,df1,cluster,mappa)
        elif typeofuser == 2:
            suggestions = favourite_contents(kmeans2,df2,cluster,mappa)
        return suggestions


    kmeans1 = joblib.load(main_folder+'kmeans_model1.pkl')
    kmeans2 = joblib.load(main_folder+'kmeans_model2.pkl')
    df1 = pd.read_csv(main_folder+'clustering_1df.csv')
    df2 = pd.read_csv(main_folder+'clustering_2df.csv')

    with open(main_folder+'map_cont.json', 'r') as file:
        map_cont = json.load(file)

    with open(main_folder+'map_cat.json', 'r') as file:
        map_cat = json.load(file)

    with open(main_folder+'map_act.json', 'r') as file:
        map_act = json.load(file)

    with open(main_folder+'map_naz.json', 'r') as file:
        map_naz = json.load(file)

    with open(main_folder+'map_occ.json', 'r') as file:
        map_occ = json.load(file)

    with open(main_folder+'map_reg.json', 'r') as file:
        map_reg = json.load(file)

    (new_group,typeofuser) = classify_utente(req,kmeans1, kmeans2, map_cat, map_naz, map_occ, map_cont, map_act, map_reg)
    suggerimenti_utente = get_suggestions(kmeans1,kmeans2,typeofuser,new_group,df1,df2,map_cont)
    print(suggerimenti_utente)
    film = np.array(suggerimenti_utente.tolist()).flatten().tolist()
    
    # Restituisci i nomi come risposta JSON
    return jsonify({"Status:":"OK", "suggestionFilms": film})



if __name__ == '__main__':
    print("Backend running at:  http://localhost:5000")
    app.run(debug=True)