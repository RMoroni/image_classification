from cnn_models import get_cnn_model_by_name
import numpy as np
from torch import nn, from_numpy, max
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

BATCH_SIZE = 85
DEVICE = "cpu"
LEARNING_RATE = 0.001
N_EPOCHS = 15
N_CLASSES = 2
WEIGHT_DECAY = 0.0001


def fit(cnn_model, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    cnn_model.to(DEVICE)

    for epoch in range(N_EPOCHS):
        x_pred_list, y_pred_list = [], []
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = cnn_model(images.float())
            loss = loss_fn(outputs, labels)

            _, predicted = max(outputs.data, 1)
            x_pred_list.append(predicted.cpu().numpy())
            y_pred_list.append(labels.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_pred_list = np.asarray(x_pred_list).ravel()
        y_pred_list = np.asarray(y_pred_list).ravel()

        acc = accuracy_score(x_pred_list, y_pred_list)

        print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {loss.item()}, Acc: {acc*100}')
    return cnn_model


def main(data, labels, cnn_model_name='default'):
    tensor_x = from_numpy(data.transpose(0, 3, 1, 2))
    tensor_y = from_numpy(labels)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cnn_model = get_cnn_model_by_name(cnn_model_name, N_CLASSES)
    fit_model = fit(cnn_model, dataloader)
    return fit_model
