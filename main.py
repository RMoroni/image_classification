import dataset
import train


def main():
    x_train, y_train, x_test, y_test = dataset.load_dataset()
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    fit_model = train.main(x_train, y_train)
    # TODO: test step


if __name__ == '__main__':
    main()
