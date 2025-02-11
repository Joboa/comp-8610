import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# XOR dataset
np.random.seed(0)
N = 2500  # Number of points per region
X = np.vstack([
    np.random.randn(N, 2) * 0.2 + [0, 0],  # Class 0
    np.random.randn(N, 2) * 0.2 + [0, 1],  # Class 1
    np.random.randn(N, 2) * 0.2 + [1, 0],  # Class 1
    np.random.randn(N, 2) * 0.2 + [1, 1],  # Class 0
])
y = np.hstack([
    np.zeros(N),   # Class 0
    np.ones(N),    # Class 1
    np.ones(N),    # Class 1
    np.zeros(N)    # Class 0
])

# Pocket Algorithm
class PocketAlgorithm:
    def __init__(self, max_iter=1000, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.best_weights = None
        self.best_accuracy = 0
    
    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.randn(X_bias.shape[1])
        self.best_weights = np.copy(self.weights)
        
        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            current_accuracy = accuracy_score(y, y_pred)
            
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_weights = np.copy(self.weights)
            
            misclassified = np.where(y_pred != y)[0]
            if len(misclassified) == 0:
                break
            
            idx = np.random.choice(misclassified)
            self.weights += self.learning_rate * (y[idx] - y_pred[idx]) * X_bias[idx]
        
        self.weights = self.best_weights
    
    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return np.where(np.dot(X_bias, self.weights) >= 0, 1, 0)

# Perceptron Algorithm
class PerceptronAlgorithm:
    def __init__(self, max_iter=1000, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.randn(X_bias.shape[1])
        
        for _ in range(self.max_iter):
            y_pred = self.predict(X)
            misclassified = np.where(y_pred != y)[0]
            if len(misclassified) == 0:
                break
            
            idx = np.random.choice(misclassified)
            self.weights += self.learning_rate * (y[idx] - y_pred[idx]) * X_bias[idx]
    
    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return np.where(np.dot(X_bias, self.weights) >= 0, 1, 0)

# Adaline
class Adaline:
    def __init__(self, max_iter=1000, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.errors_ = []
    
    def activation(self, X):
        return np.dot(X, self.weights)
    
    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.randn(X_bias.shape[1])
        
        for _ in range(self.max_iter):
            net_input = self.activation(X_bias)
            errors = y - net_input
            self.weights += self.learning_rate * np.dot(X_bias.T, errors)
            self.errors_.append(np.mean(errors**2))
            
            if len(self.errors_) > 1 and abs(self.errors_[-1] - self.errors_[-2]) < 1e-7:
                break
    
    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return np.where(self.activation(X_bias) >= 0.0, 1, 0)

# Sigmoid Model
class SigmoidNeuron:
    def __init__(self, max_iter=1000, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.errors_ = []
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.random.randn(X_bias.shape[1])
        
        for _ in range(self.max_iter):
            output = self.sigmoid(np.dot(X_bias, self.weights))
            errors = y - output
            self.errors_.append(np.mean(errors**2))
            gradient = np.dot(X_bias.T, errors * output * (1 - output))
            self.weights += self.learning_rate * gradient
            
            if len(self.errors_) > 1 and abs(self.errors_[-1] - self.errors_[-2]) < 1e-7:
                break
    
    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return np.where(self.sigmoid(np.dot(X_bias, self.weights)) >= 0.5, 1, 0)

# Model Evaluation
def evaluate_models(X, y, train_splits, n_runs=10):
    results = {
        'Pocket': [],
        'Perceptron': [],
        'Adaline': [],
        'Sigmoid': []
    }
    
    for split in train_splits:
        split_results = {model: [] for model in results.keys()}
        
        for _ in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=None)
            
            # Pocket Algorithm
            pocket = PocketAlgorithm(max_iter=1000, learning_rate=0.1)
            pocket.fit(X_train, y_train)
            split_results['Pocket'].append(accuracy_score(y_test, pocket.predict(X_test)))
            
            # Perceptron Algorithm
            perceptron = PerceptronAlgorithm(max_iter=1000, learning_rate=0.1)
            perceptron.fit(X_train, y_train)
            split_results['Perceptron'].append(accuracy_score(y_test, perceptron.predict(X_test)))
            
            # Adaline
            adaline = Adaline(max_iter=1000, learning_rate=0.01)
            adaline.fit(X_train, y_train)
            split_results['Adaline'].append(accuracy_score(y_test, adaline.predict(X_test)))
            
            # Sigmoid
            sigmoid = SigmoidNeuron(max_iter=1000, learning_rate=0.1)
            sigmoid.fit(X_train, y_train)
            split_results['Sigmoid'].append(accuracy_score(y_test, sigmoid.predict(X_test)))
        
        for model in results.keys():
            results[model].append(np.mean(split_results[model]))
    
    return results

# Evaluation
train_splits = [0.5, 0.6, 0.7, 0.8]
results = evaluate_models(X, y, train_splits)

# Decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# XOR Data Points
axes[0].scatter(X[y==0][:, 0], X[y==0][:, 1], color='red', label='Class -1', alpha=0.5)
axes[0].scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', label='Class +1', alpha=0.5)
axes[0].set_title('XOR Data Points')
axes[0].legend()

# Accuracy vs Sample Size
for model, accuracies in results.items():
    axes[1].plot(np.array(train_splits)*100, accuracies, label=model, marker='o')

axes[1].set_xlabel('Training Data Size (%)')
axes[1].set_ylabel('Average Accuracy')
axes[1].set_title('Model Accuracy vs Training Data Size')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()