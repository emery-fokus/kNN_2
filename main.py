from knn import knn
from data_loader import load_normalized_data

if __name__ == "__main__":
    # 1. Chargement des données
    X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
    
    # 2. Séparation manuelle (80% entraînement, 20% test)
    limite = int(len(X_normalized) * 0.8)
    
    # Extraction et conversion DIRECTE en listes/tableaux pour réinitialiser les index à 0
    X_train = X_normalized[:limite].values if hasattr(X_normalized, 'values') else X_normalized[:limite]
    X_test = X_normalized[limite:].values if hasattr(X_normalized, 'values') else X_normalized[limite:]
    
    # Pour Y, on transforme en liste pure 
    y_train = list(Y[:limite])
    y_test = list(Y[limite:])
    
    # 3. Initialisation du modèle
    knn_object = knn(k=3)
    
    # 4. Entraînement 
    knn_object.fit(X_train, y_train)
    
    # 5. ÉVALUATION
    print("Calcul des prédictions en cours (cela peut prendre quelques secondes)...")
    precision = knn_object.evaluate(X_test, y_test)
    
    # 6. Affichage du résultat
    print("-" * 30)
    print(f"Entraînement terminé sur {len(X_train)} échantillons.")
    print(f"Test effectué sur {len(X_test)} échantillons.")
    print(f"Précision du modèle KNN : {precision * 100:.2f} %")