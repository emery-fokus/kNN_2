from knn import KNN
from Evualator import Evaluator
class GridSearch:
    def __init__(self, k_values):
        self.k_values = k_values

    def fit_best_k(self, X_train, y_train, X_test, y_test):
        # testage de toutes les valeurs de k
        best_k = None
        best_score = 0.0
        evaluator = Evaluator()
        for k in self.k_values:
            model = KNN(k=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = evaluator.accuracy(y_test, y_pred)
            print(f"Pour k={k}, Accuracy={score*100}%")
            if score > best_score:
                best_score = score
                best_k = k

        return best_k, best_score

if __name__ == "__main__":    # Exemple d'utilisation
    X_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    X_test = [[1.5], [2.5]]
    y_test = [0, 1]

    k_values = [1, 2, 3]
    grid_search = GridSearch(k_values)
    best_k, best_score = grid_search.fit_best_k(X_train, y_train, X_test, y_test)
    print(f"Meilleur k: {best_k} avec une accuracy de {best_score*100}%")