# coding: utf-8

import pickle
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def mobile_tech_binary_classifier(inference_data):
    vect_path = PROJECT_ROOT / 'models' / 'article_vect.pkl'
    classf_path = PROJECT_ROOT / 'models' / 'article_classf.pkl'

    if os.path.exists(vect_path) and os.path.exists(classf_path):
        vectorizer = pickle.load(open(vect_path, 'rb'))
        classifier = pickle.load(open(classf_path, 'rb'))
        data = vectorizer.transform(inference_data['Text'])
        y_pred = classifier.predict(data)
        inference_data['Mobile_Tech'] = y_pred
    else:
        print(f"Warning: Article classifier models not found at {vect_path}. Defaulting Mobile_Tech predictions to 0 before brand rule.")
        inference_data['Mobile_Tech'] = 0

    options = [['google'],['microsoft'],['asus'],['hp']]
    for option in options:
        indexes = inference_data.index[inference_data['brands'].apply(lambda x:x == option)]
        inference_data = inference_data.drop(indexes,axis = 0)
    inference_data.loc[inference_data['num_brands']>0, 'Mobile_Tech'] = 1
    return inference_data

