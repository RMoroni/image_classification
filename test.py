import numpy as np
from torch import max
from sklearn.metrics import accuracy_score


def eval_neural_network_model(neural_network_eval_params: dict):
    fit_model = neural_network_eval_params['fit_model']
    dataset = neural_network_eval_params['dataset']

    fit_model.eval()

    x_pred_list, y_pred_list = [], []
    for i, (image, label) in enumerate(dataset):
        image = image.to(neural_network_eval_params['device'])
        label = label.to(neural_network_eval_params['device'])
        output = fit_model(image.float())
        # loss = loss_fn(outputs, labels)
        _, predicted = max(output.data, 1)
        x_pred_list.append(predicted.cpu().numpy())
        y_pred_list.append(label.cpu().numpy())

    x_pred_list = np.asarray(x_pred_list).ravel()
    y_pred_list = np.asarray(y_pred_list).ravel()

    acc = accuracy_score(x_pred_list, y_pred_list)

    print(f'Test Acc: {acc * 100}')

