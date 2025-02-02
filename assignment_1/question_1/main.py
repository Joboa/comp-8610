import numpy as np
import torch
import matplotlib.pyplot as plt

from config import CONFIG
from utils import synthetic_data, set_seed
from model import NNModel
from train import train_model_gd, train_model_bgd, train_model_sgd
from test_model import cross_validation, test_model


set_seed(CONFIG["seed"])
# ### (a) ####
x: np.ndarray = synthetic_data(mu=0, sigma=1, N=5000)

# ### (b) ####
# sigma = sqrt(variance) , var=0.25
eps: np.ndarray = synthetic_data(mu=0, sigma=0.5, N=5000)

# ### (c) ####
y: np.ndarray = 1 - (0.5*x) + (2*x**2) - (0.3*x**3) + eps

# plot_data(x, y, 10, 8)
# Torch: x, y
X = torch.from_numpy(x).view(-1, 1).float()
Y = torch.from_numpy(y).view(-1, 1).float()

activations = ["adaline", "sigmoid"]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
average_loss = {}

# plt.figure(figsize=(8, 6))

# for activation in activations:
#     for lr in learning_rates:
#         model = NNModel(activation)

#         losses = train_model_sgd(
#             x_train=X,
#             target=Y,
#             model=model,
#             learning_rate=lr,
#             n_epochs=50
#         )

#         plt.plot(range(len(losses)), losses, label=f"{activation}, LR={lr}")

#         avg_loss = np.mean(losses)
#         average_loss[f"{activation}_{lr}"] = avg_loss

# plt.title("SGD Loss curve for adaline/sigmoid")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("sgd_loss_plot.png", dpi=400)

# print("Average losses")
# with open("sgd_average_losses.txt", "w") as f:
#     for key, value in average_loss.items():
#         # print(f"{key}: {value:4f}")
#         f.write(f"{key}: {value:4f}\n")


##### Cross validation #####


# Train and test using cross-validation
x_train, x_test, y_train, y_test = cross_validation(X, Y, 32)

model = NNModel("sigmoid")
losses = train_model_bgd(
    x_train=x_train,
    target=y_train,
    model=model,
    learning_rate=0.001,
    n_epochs=50
)

accuracy = test_model(x_test, y_test, model)
print("accuracy: ", accuracy)
plt.plot(range(len(losses)), losses)
plt.show()