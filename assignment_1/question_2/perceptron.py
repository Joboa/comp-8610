import torch
import numpy as np

class PerceptronN():
    def __init__(self, num_features, lr=0.01):
        self.num_features = num_features
        self.learning_rate = lr
        self.weights = np.zeros((num_features, 1), dtype=float)

    def forward(self, x):
        linear = np.dot(x, self.weights)
        # predictions = np.where(linear > 0., 1., 0.)
        predictions = np.sign(linear)
        return predictions # returns 0 or 1
        
    def backward(self, x, y):  
        predictions = self.forward(x)
        errors = y - predictions
        return errors
        
    def train(self, x, y, epochs):
        print("Started training")
        loss = []
        for e in range(epochs):
            
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += self.learning_rate * (errors * x[i]).reshape(self.num_features, 1)

                print(f"Epochs: {e+1}, Loss: {errors}, Weight: {self.weights}")
                loss.append(errors)
        return loss
        
                
    def evaluate_model(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy

# class PerceptronT():
#     def __init__(self, num_features, lr=0.01):
#         self.num_features = num_features
#         self.learning_rate = lr
#         self.weights = torch.zeros(num_features, 1, dtype=float)

#         # Initialize one and zero for the perceptron output
#         self.ones = torch.ones(1)
#         self.zeros = torch.zeros(1)

#     def forward(self, x):
#         linear = torch.mm(x, self.weights)
#         predictions = torch.where(linear > 0., self.ones, self.zeros)
#         return predictions
        
#     def backward(self, x, y):  
#         predictions = self.forward(x)
#         errors = y - predictions
#         return errors
        
#     def train(self, x, y, epochs):
#         for e in range(epochs):
            
#             for i in range(y.shape[0]):
#                 errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
#                 self.weights += self.learning_rate*(errors * x[i]).reshape(self.num_features, 1)

#             print(f"Epochs: {e+1}, Loss: {errors}, Weight: {self.weights}")
      
                
#     def evaluate(self, x, y):
#         predictions = self.forward(x).reshape(-1)
#         accuracy = torch.sum(predictions == y).float() / y.shape[0]
#         return accuracy