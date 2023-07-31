from dataset import load_dataset
from model import get_model_by_name

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset[2].shape)
    model = get_model_by_name()
