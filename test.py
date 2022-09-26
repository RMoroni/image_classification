import dataset
import numpy as np
from torch import max
from sklearn.metrics import accuracy_score

DEVICE = "cpu"


def main(fit_model, data, labels):
    fit_model.eval()

    dataloader = dataset.dataset_to_dataloader(data, labels)

    x_pred_list, y_pred_list = [], []
    for i, (image, label) in enumerate(dataloader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        output = fit_model(image.float())
        # loss = loss_fn(outputs, labels)
        _, predicted = max(output.data, 1)
        x_pred_list.append(predicted.cpu().numpy())
        y_pred_list.append(label.cpu().numpy())

    x_pred_list = np.asarray(x_pred_list).ravel()
    y_pred_list = np.asarray(y_pred_list).ravel()

    acc = accuracy_score(x_pred_list, y_pred_list)

    print(f'Test Acc: {acc * 100}')

