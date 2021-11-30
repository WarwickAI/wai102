import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor
train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=ToTensor())

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Create a grid of 12 examples
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx] # Get an examples img and respective label
    # Plot each image captioned by its respective label
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(0), cmap="gray")

plt.show()

from torch import nn


# Define 1-layer network
class OneLayerNN(nn.Module):
    def __init__(self):
        super(OneLayerNN, self).__init__()
        self.flatten = nn.Flatten()  # Flatten 28x28 image to one           dimensional input
        self.linear_a = nn.Linear(28 * 28, 40)
        self.linear_b = nn.Linear(40, 10)

    def forward(self, input):
        x = self.flatten(input)
        x = self.linear_a(x)
        x = self.linear_b(x)
        return x


model = OneLayerNN()
lr = 1e-3
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


def test_loop(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


import math


# Function to train our model
def train_model(dataloader, model, epochs, loss_fn, optimizer):
    # Variables for visualising loss over time
    y_loss = []
    y_test_loss = []

    size = len(dataloader.dataset)
    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        for batch, (images, labels) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(images)
            loss = loss_fn(pred, labels)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(images)
                y_loss.append(loss)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        test_loss = test_loop(test_loader, model, loss_fn)
        y_test_loss.append(test_loss)

    # Plot loss over time
    plt.plot(range(len(y_loss)), y_loss)
    plt.plot([x * (math.floor(len(dataloader) / 100) + 1) for x in range(epochs)], y_test_loss)
    plt.title('Loss over time')
    plt.xlabel('Epoch_batch')
    plt.ylabel('Loss')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    plt.show()


# Load model if chosen, otherwise train
load = False
model_path = './models/model_weights.pth'
if not load:

    # Train and save model
    train_model(train_loader, model, epochs, loss_fn, optimizer)
    torch.save(model.state_dict(), model_path)

else:
    model.load_state_dict(torch.load(model_path))


# Test model on some examples
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
    img, label = test_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)

    pred = model(img)
    plt.title(torch.argmax(pred).numpy())
    plt.axis("off")
    plt.imshow(img.squeeze(0), cmap="gray")

plt.show()


def generate_confusion_matrix(test_dataloader, model):
    confusion_matrix = np.zeros((10, 10))
    with torch.no_grad():
        for images, labels in test_dataloader:
            preds = model(images)
            labels = labels

            for x in range(len(preds)):
                pred = torch.argmax(preds[x]).numpy()
                confusion_matrix[labels[x]][pred] += 1

    plot_confusion_matrix(confusion_matrix)


def plot_confusion_matrix(confusion_matrix):
    # Absolute predictions
    fig, ax = plt.subplots(1)

    ax.set_title('No. Predictions Confusion Matrix')
    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    plt.show()

    # Percentage predictions
    for i in range(10):
        totalPredicted = sum(confusion_matrix[:, i])
        if totalPredicted == 0:
            print("a")
            continue

        confusion_matrix[:, i] = confusion_matrix[:, i] / totalPredicted

    fig, ax = plt.subplots(1)

    ax.set_title('Percentage Predictions Confusion Matrix')
    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    plt.show()


generate_confusion_matrix(test_loader, model)
