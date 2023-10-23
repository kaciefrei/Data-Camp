import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Créez le répertoire 'uploads' s'il n'existe pas
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@st.cache_data
def load_saved_model(model_path):
    return load_model(model_path)

# Fonction de prédiction
def predict_eye_cancer(image_path, saved_model):
    # Définir les dimensions de l'image
    image_size = (224, 224)

    # Charger l'image que vous souhaitez classer
    img = image.load_img(image_path, target_size=image_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normaliser l'image

    # Faire une prédiction sur l'image avec les trois classes les plus probables
    top_k = 3  # Nombre de prédictions à obtenir
    predictions = saved_model.predict(img)
    top_classes = np.argsort(predictions[0])[-top_k:][::-1]
    train_data_dir = r'C:\Users\kacib\Desktop\EFREI\Semestre 7\Data Camp\dataset\Training_Set\Training_Set\class'
    class_labels = os.listdir(train_data_dir)

    # Créez une liste pour stocker les résultats de prédiction
    prediction_results = []

    # Ajoutez les trois prédictions les plus probables avec les pourcentages de fiabilité à la liste
    for i, class_index in enumerate(top_classes):
        class_label = class_labels[class_index]
        confidence = predictions[0][class_index] * 100.0
        prediction_results.append({
            "class_label": class_label,
            "confidence": confidence
        })

    # Renvoyez les résultats de prédiction
    return prediction_results

def main():
    st.title('Prédiction de cancers des yeux')

    uploaded_file = st.file_uploader("Veuillez télécharger votre image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_path = os.path.join('uploads', uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Charger le modèle sauvegardé
        model_path = 'modele_classification_resnet101.h5'
        saved_model = load_saved_model(model_path)

        # Effectuer la prédiction
        prediction = predict_eye_cancer(image_path, saved_model)

        # Afficher les résultats de prédiction
        st.subheader('Résultats de l\'analyse')
        for i, pred in enumerate(prediction):
            st.write(f"Prédiction {i + 1}: {pred['class_label']}, Fiabilité : {round(pred['confidence'], 2)}%")

        # Supprimez le fichier image téléchargé (si nécessaire)
        os.remove(image_path)

if __name__ == '__main__':
    main()
