import numpy as np
from torch import nn, max
from torch.optim import Adam
from sklearn.metrics import accuracy_score


def fit(neural_network_fit_params: dict):
    nn_model = neural_network_fit_params['neural_network_model']
    dataset = neural_network_fit_params['dataset']
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(nn_model.parameters(), lr=neural_network_fit_params['learning_rate'],
                     weight_decay=neural_network_fit_params['weight_decay'])
    nn_model.to(neural_network_fit_params['device'])

    for epoch in range(neural_network_fit_params['number_of_epochs']):
        x_pred_list, y_pred_list = [], []
        for i, (images, labels) in enumerate(dataset):
            images = images.to(neural_network_fit_params['device'])
            labels = labels.to(neural_network_fit_params['device'])
            outputs = nn_model(images.float())
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

        print(f"Epoch [{epoch + 1}/{neural_network_fit_params['number_of_epochs']}], Loss: {loss.item()}, Acc: {acc * 100}")
    return nn_model
