from torch import nn, from_numpy, no_grad, max
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 85
DEVICE = "cpu"
INPUT_LENGTH = 320 * 370 * 3  # image_size * channels
LEARNING_RATE = 0.001
N_EPOCHS = 5
N_CLASSES = 2


def convolutional_neural_network_model():
    cnn_model = nn.Sequential(
        ## ConvBlock 1
        nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        ## ConvBlock 2
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        ## ConvBlock 3
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.Tanh(),
        nn.Flatten(),

        ## DenseBlock
        nn.Linear(198560, 84),
        nn.Tanh(),
        nn.Linear(84, N_CLASSES),
    )
    return cnn_model


def fit(cnn_model, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(cnn_model.parameters(), lr=0.001, weight_decay=0.0001)
    cnn_model.to(DEVICE)
    # We use the pre-defined number of epochs to determine how many iterations to train the network on
    for epoch in range(N_EPOCHS):
        # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(dataloader):
            # Move tensors to the configured device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = cnn_model(images.float())
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, N_EPOCHS, loss.item()))

    with no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = cnn_model(images.float())
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))


def main(data, labels):
    tensor_x = from_numpy(data.transpose(0, 3, 1, 2))
    tensor_y = from_numpy(labels)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cnn_model = convolutional_neural_network_model()
    fit(cnn_model, dataloader)
