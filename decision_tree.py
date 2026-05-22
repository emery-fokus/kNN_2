
# ARBRE DE DECISION 
def separer_donnees(X, y, index_colonne, seuil):
    # Listes vides pour séparer les trucs à gauche et à droite
    X_g, y_g = [], []
    X_d, y_d = [], []

    # On boucle sur toutes les lignes pour faire le tri
    for i in range(len(X)):
        valeur = X[i][index_colonne]
        
        # Plus petit ou égal -> ça part à gauche
        if valeur <= seuil:
            X_g.append(X[i])
            y_g.append(y[i])
        else:
            # Sinon ça va à droite
            X_d.append(X[i])
            y_d.append(y[i])

    return X_g, y_g, X_d, y_d


def calculer_gini(y):
    n = len(y)
    
    # Sécurité au cas où le groupe est vide pour pas crash
    if n == 0:
        return 0.0

    # Je compte combien de fois on voit chaque classe (0, 1 ou 2)
    dico_comptes = {}
    for classe in y:
        if classe in dico_comptes:
            dico_comptes[classe] += 1
        else:
            dico_comptes[classe] = 1


    somme_carres = 0.0
    for classe in dico_comptes:
        p = dico_comptes[classe] / n
        somme_carres += p * p

    return 1.0 - somme_carres


class Noeud:
    def __init__(self, attribut=None, seuil=None, gauche=None, droit=None, valeur_feuille=None):
        # Paramètres pour les questions au milieu de l'arbre
        self.attribut = attribut  # le numéro de la colonne testée
        self.seuil = seuil        # la valeur limite
        self.gauche = gauche      # la suite si c'est OUI
        self.droit = droit        # la suite si c'est NON
        
        # Si c'est une feuille tout en bas, on stocke juste la réponse
        self.valeur_feuille = valeur_feuille

    def est_une_feuille(self):
        # Si y'a une valeur de feuille, c'est que c'est la fin du voyage
        return self.valeur_feuille is not None


class ArbreDecision:
    def __init__(self, profondeur_max=5):
        self.profondeur_max = profondeur_max
        self.racine = None  # va contenir le tout premier noeud du haut

    def fit(self, X, y):
        # Lance la grosse fonction récursive à la profondeur 0
        self.racine = self._construire_arbre(X, y, 0)

    def _construire_arbre(self, X, y, prof):
        nb_lignes = len(X)
        
        # Autre sécurité anti-bug
        if nb_lignes == 0:
            return None

        nb_colonnes = len(X[0])
        liste_classes = list(set(y))

        # --- CONDITION D'ARRET ---
        if prof >= self.profondeur_max or len(liste_classes) == 1 or nb_lignes < 2:
            # classe qui revient le plus souvent
            cpt = {}
            for c in y:
                cpt[c] = cpt.get(c, 0) + 1
            majorite = max(cpt, key=cpt.get)
            return Noeud(valeur_feuille=majorite)

        # --- CHERCHER LA MEILLEURE DIVISION ---
        meilleur_col = None
        meilleur_seuil = None
        meilleur_gini = 999.0  

        # teste  des colonnes une par une
        for col in range(nb_colonnes):
            # Je récupère toutes les valeurs possibles de cette colonne pour les tester
            valeurs = []
            for ligne in X:
                valeurs.append(ligne[col])
            tous_les_seuils = list(set(valeurs))

            #  teste chaque valeur comme un seuil de coupure
            for s in tous_les_seuils:
                _, y_g, _, y_d = separer_donnees(X, y, col, s)

            
                if len(y_g) == 0 or len(y_d) == 0:
                    continue

                # Calcul du Gini moyen des enfants avec les proportions
                p_g = len(y_g) / nb_lignes
                p_d = len(y_d) / nb_lignes
                gini_total = (p_g * calculer_gini(y_g)) + (p_d * calculer_gini(y_d))

                if gini_total < meilleur_gini:
                    meilleur_gini = gini_total
                    meilleur_col = col
                    meilleur_seuil = s

        # --- CREATION DES BRANCHES ---
        if meilleur_col is None:
            cpt = {}
            for c in y:
                cpt[c] = cpt.get(c, 0) + 1
            majorite = max(cpt, key=cpt.get)
            return Noeud(valeur_feuille=majorite)

        # séparation avec notre gagnant
        X_g, y_g, X_d, y_d = separer_donnees(X, y, meilleur_col, meilleur_seuil)

        #  relance la fonction pour la gauche et la droite en faisant prof + 1
        fils_gauche = self._construire_arbre(X_g, y_g, prof + 1)
        fils_droit = self._construire_arbre(X_d, y_d, prof + 1)

        # enfants sur le noeud actuel
        return Noeud(attribut=meilleur_col, seuil=meilleur_seuil, gauche=fils_gauche, droit=fils_droit)

    def predict_one_point(self, x, noeud):
      
        if noeud.est_une_feuille():
            return noeud.valeur_feuille

     
        valeur = x[noeud.attribut]

        if valeur <= noeud.seuil:
            return self.predict_one_point(x, noeud.gauche)
        else:
            return self.predict_one_point(x, noeud.droit)

    def predict(self, X):
        res = []
        for ligne in X:
         
            p = self.predict_one_point(ligne, self.racine)
            res.append(p)
        return res