import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

t0 = time.time()
# Define the domain and create the grid
N = 20  # Number of points along each axis
x = torch.linspace(0, 1, N)
y = torch.linspace(0, 1, N)
x, y = torch.meshgrid(x, y)

# intialize the error record
loss_list = []
error_list = []

# Create input data (x, y coordinates) and target data (true solution values)
xy = torch.stack([x.flatten(), y.flatten()], dim=1)
true_solution = -(torch.sin(np.pi * x) * torch.sin(np.pi * y)) / (2 * np.pi**2)
true_solution = true_solution.flatten()

# Neural network architecture
class PoissonSolver(nn.Module):
    def __init__(self):
        super(PoissonSolver, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the model, loss function, and optimizer
model = PoissonSolver()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the exponential learning rate scheduler
gamma = 0.999  # Exponential decay factor
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Convert data to PyTorch DataLoader
dataset = TensorDataset(xy, true_solution)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Count the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params}")

# Training loop
epochs = 300
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        # print(inputs.numpy().shape, labels.numpy().shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.flatten(), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate after each epoch
        running_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # Record the loss and error
    loss_list.append(running_loss / len(dataloader))
    # with torch.no_grad():
    #     predicted_solution = model(xy)
    # error = np.linalg.norm(predicted_solution - true_solution)
    # error_list.append(error)
    # print(error)

# Evaluate the trained model on the whole domain
with torch.no_grad():
    predicted_solution = model(xy).reshape(N, N)

# Display the results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# X, Y = np.meshgrid(x.numpy(), y.numpy())
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, predicted_solution.numpy(), cmap='viridis')
# ax.set_title('Predicted Solution')
# plt.show()

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, true_solution.reshape(N,N).numpy(), cmap='viridis', label='True solution', alpha = 1)
ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(x, t, U, cmap='viridis')
ax.plot_surface(x, y, predicted_solution.numpy(), cmap='plasma', label='Predicted Solution', alpha = 1)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Plot of the true/numerical solution')
# ax.legend()

# Show the plot
plt.savefig('plotmlpsolver-v2.png')


# Plot the training loss over epochs
fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
plt.plot(range(1, epochs + 1), loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid()
plt.savefig('trainingloss_mlpsolver-v2.png')


# Plot the log of training loss over epochs
fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
plt.plot(range(1, epochs + 1), -np.log(loss_list), label='Log of Training Loss')
plt.xlabel('Epoch')
plt.ylabel('-Log(Loss)')
plt.title('Log Training Loss over Epochs')
plt.legend()
plt.grid()
plt.savefig('logtrainingloss_mlpsolver-v2.png')

err = np.linalg.norm(true_solution.reshape(N,N)-predicted_solution)/N
print("Error:",err)

# Plot the function error over epochs
# fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
# plt.plot(range(1, epochs + 1), error_list, label='Error from true solution')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.title('Error over Epochs')
# plt.legend()
# plt.grid()
# plt.savefig('errorcurve_mlpsolver-v2.png')

# Plot the -log of the error over epochs
# fig = plt.figure()
# plt.plot(range(100, epochs + 1), error_list[99::], label='Training Loss')
# plt.plot(range(1, epochs + 1), -np.log(error_list), label='-Log of Error')
# plt.xlabel('Epoch')
# plt.ylabel('-Log Error')
# plt.title('-Log of Error over Epochs')
# plt.legend()
# plt.grid()
# plt.savefig('logerrorcurve_mlpsolver-v2.png')

# record the time
t1 = time.time()
print("Running time:", t1-t0)