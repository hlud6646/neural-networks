# %% Imports and configuration
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, mean_squared_error
from typing import Callable, Optional

sns.set_palette("muted")
colors = sns.color_palette()
sns.set_style("darkgrid")
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["figure.edgecolor"] = "none"
plt.rcParams["figure.figsize"] = (6, 6)


# %% Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


# %% Model
class Model:

    def __init__(
        self,
        hidden_size: int,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        hidden_activation: Callable = np.tanh,
        output_activation: Callable = sigmoid,
        scoring: Optional[Callable] = None,
    ):
        """Initialize neural network model.

        Args:
            hidden_size: Number of neurons in hidden layer
            learning_rate: Learning rate for gradient descent
            batch_size: Number of samples per batch
            hidden_activation: Activation function for hidden layer
            output_activation: Activation function for output layer
            scoring: Optional scoring function for tracking progress
        """
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.batch_size = batch_size
        self.scoring = scoring
        self.hidden_activation = hidden_activation
        self.hidden_activation_grad = egrad(hidden_activation)
        self.output_activation = output_activation
        self.output_activation_grad = egrad(output_activation)

    def init_weights_and_biases(self, x, y):
        """
        Initialize weights and biases with small random numbers.
        """
        input_size = x.shape[1]
        hidden_size = self.hidden_size
        output_size = 1 if len(y.shape) == 1 else y.shape[1]
        rand = lambda dims: np.random.randn(*dims) / 100
        self.w1 = rand((hidden_size, input_size))
        self.b1 = rand((hidden_size,))
        self.w2 = rand((output_size, hidden_size))
        self.b2 = rand((output_size,))

    def forward(self, x):
        """
        Forward pass through the network.
        """
        self.z1 = self.w1 @ x + self.b1
        self.a1 = self.hidden_activation(self.z1)
        self.z2 = self.w2 @ self.a1 + self.b2
        self.a2 = self.output_activation(self.z2)

    def backward(self, x, y):
        """
        Backward pass through the network.
        """
        if self.output_activation == softmax:
            # Special handling for softmax + cross-entropy
            d2 = self.a2 - y  # Direct gradient for softmax with cross-entropy
        else:
            # Original handling for sigmoid/linear
            d2 = (self.a2 - y) * self.output_activation_grad(self.z2)

        d1 = (self.w2.T @ d2) * self.hidden_activation_grad(self.z1)
        # Gradient of loss wrt w1, b1, w2, b2
        self.cost_gradients = [d1, x * d1[:, None], d2, self.a1 * d2[:, None]]

    def batch(self, x, y):
        grads = []
        for xi, yi in zip(x, y):
            self.forward(xi)
            self.backward(xi, yi)
            grads.append(self.cost_gradients)
        grads = [np.stack(g, axis=0) for g in list(zip(*grads))]
        db1, dw1, db2, dw2 = [a.mean(axis=0) for a in grads]

        # Gradient descent.
        self.b1 -= self.lr * db1
        self.w1 -= self.lr * dw1
        self.b2 -= self.lr * db2
        self.w2 -= self.lr * dw2

    def run_epoch(self, x, y):
        for i in range(0, len(x), self.batch_size):
            batch_x = x[i : i + self.batch_size]
            batch_y = y[i : i + self.batch_size]
            self.batch(batch_x, batch_y)

    def fit(self, x, y, n_epochs=100):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.init_weights_and_biases(x, y)
        history = []
        start_time = time.time()
        for epoch in range(n_epochs):
            self.run_epoch(x, y)
            if self.scoring:
                history.append(self.scoring(y, self.predict(x)))
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:>3d}  |  Loss: {history[-1]:.4f}  |  Time: {elapsed:.1f}s"
                )
            else:
                print(f"Epoch {epoch:>3d}  |  Time: {elapsed:.1f}s")
        return history or None

    def predict_one(self, xi):
        self.forward(xi)
        return self.a2

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        predictions = np.array([self.predict_one(xi) for xi in x])
        if predictions.shape[1] == 1:
            predictions = predictions.reshape(-1)
        return predictions


# %% Binary classification demonstration

x, y = load_breast_cancer(as_frame=True, return_X_y=True)
x = x[
    [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "worst radius",
        "worst texture",
        "worst perimeter",
    ]
]
x = (x - x.mean(axis=0)) / x.std(axis=0)

m = Model(
    hidden_size=3,
    learning_rate=0.05,
    batch_size=32,
    hidden_activation=np.tanh,
    output_activation=sigmoid,
    scoring=log_loss,
)

history = m.fit(x, y, n_epochs=50)

yhat = m.predict(x)
print(f"Accuracy: {np.mean(y == (yhat > 0.5)):.2%}")
correct = y == (yhat > 0.5)
correct = np.where(correct, "Correct", "Incorrect")

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
axs[0].plot(history)
axs[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
sns.scatterplot(x=range(len(yhat)), y=yhat, hue=correct, style=correct, s=20)
axs[1].legend(loc="center right")
axs[1].set(title="Hit or Miss", xlabel="Index", ylabel="Prediction")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join("..", "out", "breast_cancer_classification.png"))

# %% Multiclass classification demo.

df = pd.read_csv(os.path.join("..", "data", "fish_data.csv"))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
x, y = df.drop(columns=["species"]), df["species"]
y = pd.get_dummies(y)
x = (x - x.mean(axis=0)) / x.std(axis=0)

m = Model(
    hidden_size=16,
    learning_rate=0.01,
    batch_size=32,
    hidden_activation=np.tanh,
    output_activation=softmax,
    scoring=log_loss,
)

history = m.fit(x, y, n_epochs=50)

yhat = np.argmax(m.predict(x), axis=1)
yv = np.argmax(y, axis=1)
print(f"Accuracy: {np.mean(yhat == yv):.2%}")

fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(yhat, yv), annot=True, fmt="d", cbar=False, ax=ax)
ax.set(xlabel="Predicted", ylabel="Actual")
# Use the column names from x as labels.
ax.set_xticklabels(y.columns, rotation=90)
ax.set_yticklabels(y.columns, rotation=0)
# plt.show()
plt.tight_layout()
plt.savefig(os.path.join("..", "out", "fish_classification.png"))

# %% Regression demo.
df = pd.read_csv(os.path.join("..", "data", "CarsData.csv"))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
x = df[["year", "mileage", "mpg", "engineSize"]]
x = (x - x.mean(axis=0)) / x.std(axis=0)
y = (df["price"] - df["price"].mean()) / df["price"].std()

m = Model(
    hidden_size=6,
    learning_rate=0.05,
    batch_size=128,
    output_activation=lambda x: x,
    scoring=mean_squared_error,
)

history = m.fit(x, y, n_epochs=5)

indices = np.random.choice(len(x), 2000)
x_sample = x.iloc[indices]
y_sample = y[indices]
y_hat_sample = m.predict(x_sample)

fig, ax = plt.subplots()
ax.scatter(y_sample, y_hat_sample, label="Predicted", s=1, alpha=0.5)
ax.set(xlabel="Actual", ylabel="Predicted", xlim=(-2, 3), ylim=(-2, 3))
# Add the diagonal line.
ax.plot([-2, 3], [-2, 3], color="black", lw=1, alpha=0.3)
# plt.show()
plt.savefig(os.path.join("..", "out", "cars_regression.png"))
