import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check for CUDA availability and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple feedforward neural network with a dynamic number of hidden layers
class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_hidden_layers):
        super(DynamicNN, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


batch_size = 1000

# Define the specified transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    torchvision.transforms.Lambda(torch.flatten)
])

# Load MNIST datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_size = 28 * 28  # MNIST image size (flattened)
hidden_size = 64    # Number of hidden units   
num_classes = 10  # Digits 0-9
num_hidden_layers = 2  # Set the number of hidden layers here


# Hyperparameters
learning_rate = 0.001
num_epochs = 30



hidden_range = range(2, 7, 1)

# calculate l2 distances 20 times
for j in range(0, 21):
    l2_distances = []
    for num_hidden_layers in hidden_range:
        print(f"Number of hidden layers: {num_hidden_layers}")
        
        # Create the model, loss function, and optimizer
        model = DynamicNN(input_size, hidden_size, num_classes, num_hidden_layers).to(device)  # Move model to CUDA
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # get initial weights
        initial_weights = []
        for param in model.state_dict():
            if 'hidden' in param and 'weight' in param:
                initial_weights.append(torch.clone(model.state_dict()[param].flatten()))
        initial_weights = torch.stack(initial_weights)
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                
                # Move images and labels to the GPU
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/i:.4f}')
            if total_loss/i < 0.1:
                break

        # End of training, get trained weights
        trained_weights = []
        for param in model.state_dict():
            if 'hidden' in param and 'weight' in param:
                trained_weights.append(model.state_dict()[param].flatten())
        trained_weights = torch.stack(trained_weights)
        
        # get average l2 distance between initial and trained weights
        l2_distance = torch.mean(torch.norm(trained_weights - initial_weights, dim=1))
        l2_distances.append(float(l2_distance))
        print(l2_distances)

    with open(f'result{j}.txt', 'w') as file:
        file.write(repr(l2_distances))