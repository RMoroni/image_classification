import numpy as np
from torch import nn, from_numpy, max
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score


def test(fit_model, data, labels):
    fit_model.eval()

    tensor_x = from_numpy(data.transpose(0, 3, 1, 2))
    tensor_y = from_numpy(labels)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset=tensor_dataset, shuffle=True)

    x_pred_list, y_pred_list = [], []

    for i, (image, label) in enumerate(dataloader):
        image = image.to("cpu")
        label = label.to("cpu")
        output = fit_model(image.float())
        # loss = loss_fn(outputs, labels)
        _, predicted = max(output.data, 1)
        x_pred_list.append(predicted.cpu().numpy())
        y_pred_list.append(label.cpu().numpy())

    x_pred_list = np.asarray(x_pred_list).ravel()
    y_pred_list = np.asarray(y_pred_list).ravel()

    acc = accuracy_score(x_pred_list, y_pred_list)

    print(f'Test Acc: {acc * 100}')

