class Evaluator:
    def __init__(self):
        pass
    def accuracy(self, y_true, y_pred):
        """Calcul de la precison"""
        if len(y_true)==0:
            return 0.0
        correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / len(y_true)
    
if __name__ == "__main__":
    evaluator = Evaluator()
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    score = evaluator.accuracy(y_true, y_pred)
    print(f"Accuracy de test:{score*100}%")
