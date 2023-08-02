from dataset import load_dataset
from model import get_model_by_name
from test import predict_and_show
from train import train_model


if __name__ == '__main__':
    dataset = load_dataset()
    model = get_model_by_name()
    train_model(model, dataset[0], dataset[1])
    predict_and_show(dataset[2], model)
