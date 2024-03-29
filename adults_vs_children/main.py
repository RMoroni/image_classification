import dataset
import train
import test
import torch
from cnn_models import get_cnn_model_by_name

BATCH_SIZE = 85
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
N_EPOCHS = 30
N_CLASSES = 2
WEIGHT_DECAY = 1e-6
# ORIGINAL IMAGE SIZE: 370x320


def main():
    x_train, y_train, x_test, y_test = dataset.load_dataset()
    train_dataset = dataset.dataset_to_dataloader(x_train, y_train, BATCH_SIZE)
    test_dataset = dataset.dataset_to_dataloader(x_test, y_test)

    cnn_model = get_cnn_model_by_name('default_color', N_CLASSES)

    neural_network_fit_params = {
        'neural_network_model': cnn_model,
        'dataset': train_dataset,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'number_of_epochs': N_EPOCHS,
        'number_of_class': N_CLASSES,
        'weight_decay': WEIGHT_DECAY,
        'device': DEVICE,
    }
    fit_model = train.fit(neural_network_fit_params)

    neural_network_eval_params = {
        'fit_model': fit_model,
        'dataset': test_dataset,
        'device': DEVICE,
    }
    test.eval_neural_network_model(neural_network_eval_params)


if __name__ == '__main__':
    main()
