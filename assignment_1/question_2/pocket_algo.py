import numpy as np

class PocketPerceptron():
    def __init__(self, num_features, lr=0.01):
        self.num_features = num_features
        self.learning_rate = lr
        self.weights = np.zeros((num_features, 1), dtype=float)
        # self.weights = np.random.normal(num_features, 1)
        self.best_weights = self.weights.copy()  
        self.best_error_count = float('inf')

    def forward(self, x):
        linear = np.dot(x, self.weights)
        predictions = np.where(linear > 0., 1., 0.)
        return predictions  # returns 0 or 1

    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors

    def train(self, x, y, epochs):
        loss = []
        for epoch in range(epochs):
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += self.learning_rate * (errors * x[i]).reshape(self.num_features, 1)
                
                # Number of misclassifications
                current_predictions = self.forward(x)
                current_error_count = np.sum(y != current_predictions) # 0 if predictions == target
                
                if current_error_count < self.best_error_count:
                    self.best_error_count = current_error_count
                    self.best_weights = self.weights.copy()

                print(f"Epochs: {epoch+1}, Loss: {errors}, Weight: {self.weights}")
                loss.append(errors)

        self.weights = self.best_weights.copy()

        return loss, self.best_weights
    
    def evaluate_model(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy
