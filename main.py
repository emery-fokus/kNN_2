from knn import knn
from decision_tree import ArbreDecision
from data_loader import load_normalized_data
from model_tools import ModelTools

if __name__ == "__main__":
    # 1. Chargement des données
    X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
    
    # 2. Séparation manuelle (80% entraînement, 20% test)
    limite = int(len(X_normalized) * 0.8)
    
    # On extrait d'abord en brut
    X_train_raw = X_normalized[:limite]
    X_test_raw = X_normalized[limite:]
    
    # --- NETTOYAGE RADICAL POUR LE KNN ET L'ARBRE ---
    # On convertit tout en listes Python de listes pour être 100% sûr d'avoir des lignes.
    # Si la ligne n'est pas une liste (un float seul), on l'enferme dans une liste : [valeur]
    X_train = []
    for ligne in X_train_raw:
        if hasattr(ligne, "tolist"): # Si c'est du numpy
            val = ligne.tolist()
            X_train.append(val if isinstance(val, list) else [val])
        elif isinstance(ligne, (list, tuple)):
            X_train.append(list(ligne))
        else:
            X_train.append([ligne])

    X_test = []
    for ligne in X_test_raw:
        if hasattr(ligne, "tolist"):
            val = ligne.tolist()
            X_test.append(val if isinstance(val, list) else [val])
        elif isinstance(ligne, (list, tuple)):
            X_test.append(list(ligne))
        else:
            X_test.append([ligne])
        
    y_train = list(Y[:limite])
    y_test = list(Y[limite:])
    
    # --- EVALUATION DU MODELE KNN ---
    print("🤖 Modele 1 : KNN")
    knn_object = knn(k=3)
    knn_object.fit(X_train, y_train)
    
    print("Calcul des prédictions KNN en cours...")
    precision_knn = ModelTools.evaluate(knn_object, X_test, y_test)
    
    # --- EVALUATION DE TON ARBRE DE DECISION ---
    print("\n🌲 Modele 2 : Arbre de Decision")
    arbre_object = ArbreDecision(profondeur_max=5)
    
    print("Entraînement de l'arbre en cours...")
    arbre_object.fit(X_train, y_train)
    
    print("Calcul des prédictions Arbre en cours...")
    precision_arbre = ModelTools.evaluate(arbre_object, X_test, y_test)
    
    # 6. Affichage du résultat final
    print("\n" + "=" * 40)
    print(f"Entraînement terminé sur {len(X_train)} échantillons.")
    print(f"Test effectué sur {len(X_test)} échantillons.")
    print("-" * 40)
    print(f"Précision du modèle KNN      : {precision_knn * 100:.2f} %")
    print(f"Précision de ton Arbre       : {precision_arbre * 100:.2f} %")
    print("=" * 40)