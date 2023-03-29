from torch import nn


def _default_gray(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # DenseBlock
        nn.Flatten(),
        nn.Linear(1024, 8),
        nn.Sigmoid(),
        # nn.Linear(256, 128),
        # nn.Sigmoid(),
        # nn.Linear(128, 64),
        # nn.Sigmoid(),
        nn.Linear(8, n_classes),
    )
    return cnn_model


def _default_color(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 4
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
        # nn.BatchNorm2d(512),
        # nn.LeakyReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # DenseBlock
        nn.Flatten(),
        nn.BatchNorm1d(1024),

        nn.Linear(1024, 512),
        nn.Sigmoid(),
        nn.Linear(512, 256),
        nn.Sigmoid(),
        nn.Linear(256, 128),
        nn.Sigmoid(),
        nn.Linear(128, n_classes),
    )
    return cnn_model  # 80 x 80


def get_cnn_model_by_name(name='default_color', n_classes=2):
    if name == 'default_gray':
        return _default_gray(n_classes)
    else:
        return _default_color(n_classes)
