import numpy as np
import torch
import matplotlib.pyplot as plt
from config import CONFIG

# from model import Adaline, AdalineBGD, trainBGD, trainSGD, loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def synthetic_data(mu: int, sigma: float, N: int) -> np.ndarray:
    """
    Generate x samples from a Guassian distribution N(mean=0, std=1)
    mu: mean, sigma: standard deviation, N = size of dataset
    """

    return np.random.normal(loc=mu, scale=sigma, size=N)

def plot_data(x, y, fig_x, fig_y):
    plt.figure(figsize=(fig_x,fig_y))
    plt.scatter(x, y)
    plt.title("Cubic function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def SGDAdaline(Input, d_train_y, lr=0.2, stop=0.001):
	weight = np.random.random(Input.shape[1])
	
	Error=[stop +1]
	# check the stop condition for the network
	while Error[-1] > stop or Error[-1]-Error[-2] > 0.0001:
		error = []
		for i in range(Input.shape[0]):
			Y_input = sum(weight*Input[i]) + bias
			
			# Update the weight
			for j in range(Input.shape[1]):
				weight[j]=weight[j] + lr*(d_train_y[i]-Y_input)*Input[i][j]

			# Update the bias
			bias=bias + lr*(d_train_y[i]-Y_input)
			
			# Store squared error value
			error.append((d_train_y[i]-Y_input)**2)
		# Store sum of square errors
		Error.append(sum(error))
		print('Error :',Error[-1])
	return weight, bias

def activation(x, act_type):
    if act_type == "adaline":
          return x
     
    if act_type == "sigmoid":
          return 1 / (1 + np.exp(-x))
          
    if act_type == "tanh":
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
     

def BGDAdaline(d_train_x, d_train_y, lr=0.2, act_type="adaline", n_epochs=10):
    n_features = 1 if len(d_train_x.shape) == 1 else d_train_x.shape[1]

    weight = np.random.random(n_features)
    
    for _ in range(n_epochs):

        # Initialize gradients
        weight_gradient = np.zeros_like(weight)

        error = []
        
        for i in range(d_train_x.shape[0]):
            net_input = sum(weight * d_train_x[i])
            y_hat = activation(net_input, act_type)

            # Accumulate gradients
            for j in range(n_features):
                weight_gradient[j] += lr *(d_train_y[i] - y_hat) * d_train_x[i]

            # Store squared error value
            mse = (d_train_y[i] - y_hat) ** 2
            print("MSE:", mse)
            error.append(mse)
        
        # Update weights using accumulated gradients
        weight += weight_gradient / d_train_x.shape[0]
    return weight, error


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    # ### (a) ####
    x: np.ndarray = synthetic_data(mu=0, sigma=1, N=10)

    # ### (b) ####
    eps: np.ndarray = synthetic_data(mu=0, sigma=0.5, N=10) # sigma = sqrt(variance) , var=0.25

    # ### (c) ####
    y: np.ndarray = 1 - (0.5*x )+ (2*x**2) - (0.3*x**3) + eps

    

    # plot_data(x, y, 10,8)

    # Torch: x, y
    # X = torch.from_numpy(x).view(-1,1).float()
    # Y = torch.from_numpy(y).view(-1,1).float()
    # # print(X.shape, Y.shape)
    # print(X.size(), Y.size)
    w, cost = BGDAdaline(x, y, lr=0.001, act_type="sigmoid", n_epochs=200)
    print("weight", w)
    print("cost", cost)
    def prediction(X,w):
        y=[]
        for i in range(X.shape[0]):
            x = X[i]
            y.append(sum(w*x))
        return y
    # print("y:", y[:5])
    # pred = prediction(x,w)
    # print("pred:", pred[:5])

    plt.plot(range(len(cost)), cost)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()
