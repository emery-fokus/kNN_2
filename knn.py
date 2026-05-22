class knn:
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict_one_point(self, x_test_point):
        distances = []
        for index in range(len(self.x_train)):
            x_train = self.x_train[index]
            sommes_carre = 0
            for index_j in range(len(x_test_point)):
                sommes_carre += (x_test_point[index_j] - x_train[index_j]) ** 2
            distance = sommes_carre ** 0.5
            distances.append((distance, self.y_train[index]))
        
        distances.sort(key=lambda couple: couple[0])
        neighbors = distances[:self.k]
      
        compteur_vote = {}
        for neighbor in neighbors:
            label = neighbor[1]
            if label not in compteur_vote:
                compteur_vote[label] = 1
            else:
                compteur_vote[label] += 1
                
        compteur_vote_gagnant = max(compteur_vote, key=compteur_vote.get)
        return compteur_vote_gagnant
   
    def predict(self, x_test):
        y_pred = []
        for x_test_point in x_test:
            y_pred.append(self.predict_one_point(x_test_point))
        return y_pred
   
    # def evaluate(self, x_test, y_test):
    #     y_pred = self.predict(x_test)
    #     compteur = 0
    #     for index in range(len(y_pred)):
    #         if y_pred[index] == y_test[index]:
    #             compteur += 1
    
    #     precision = compteur / len(y_test)
    #     return precision