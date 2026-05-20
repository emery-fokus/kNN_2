class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, Y):
        """Mémorisation des données d'entraînement"""
        self.X_train = X
        self.y_train = Y

    def distance(self, a, b):
        """Calcul de la distance euclidienne """
        total = 0
        for i in range(len(a)):
            total += (a[i] - b[i]) ** 2
        return total ** 0.5

    def _predict_one(self, x):
        """Prédiction pour UN SEUL échantillon"""
        distances = []
        for i in range(len(self.X_train)):
            dist = self.distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_voisins = distances[:self.k]
        
        # Vote majoritaire
        votes = {}
        for dist, label in k_voisins:
            if label not in votes:
                votes[label] = 1
            else:
                votes[label] += 1
                
        label_pred = max(votes, key=votes.get)
        return label_pred

    def predict(self, X):
        """Prédiction pour une LISTE d'échantillons"""
        predictions = []
        for x in X:
            predictions.append(self._predict_one(x))
        return predictions


   



if __name__ == "__main__":
    model = KNN(k=3)
    X = [[0], [1], [2], [3]]
    Y = [0, 0, 1, 1]
    model.fit(X, Y)
    print("Test unitaire [2]  :", model._predict_one([2]))   
    print("Test unitaire [11] :", model._predict_one([11]))  
    print("Test liste globale :", model.predict([[0], [3]])) 
