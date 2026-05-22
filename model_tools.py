
class ModelTools:
    @staticmethod
    def evaluate(model, x_test, y_test):
        """
        Calcule la précision (accuracy) de n'importe quel modèle 
        qui possède une méthode .predict()
        """
        y_pred = model.predict(x_test)
        compteur = 0
        
        for index in range(len(y_pred)):
            if y_pred[index] == y_test[index]:
                compteur += 1
                
        precision = compteur / len(y_test)
        return precision

    @staticmethod
    def grid_search(model_class, param_grid, x_train, y_train, x_val, y_val):
        """
        Teste automatiquement différentes combinaisons de paramètres 
        pour trouver la meilleure.
        """
        best_score = -1
        best_params = {}
       
        pass

