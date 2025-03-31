from nn import MLP
from optim import SDG
from autograd import Tanh

mlp = MLP(3, [4, 4, 1], activation_fn=Tanh())

x_train = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

y_train = [1.0, -1.0, -1.0, 1.0]

def mse_loss(y_true, y_pred):
    return sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

epochs = 20
learning_rate = 0.1
optimizer = SDG(params=mlp.params(), learning_rate=learning_rate)

# training loop
for epoch in range(epochs):
    y_pred = [mlp(x)[0] for x in x_train]
    optimizer.zero_grad()
    loss = mse_loss(y_train, y_pred)
    loss.backward()
    optimizer.step()

    print(f"epoch: {epoch} | loss: {loss.val}")