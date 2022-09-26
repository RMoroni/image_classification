import dataset
import train
import test


def main():
    x_train, y_train, x_test, y_test = dataset.load_dataset()
    fit_model = train.main(x_train, y_train)
    test.test(fit_model, x_test, y_test)


if __name__ == '__main__':
    main()
