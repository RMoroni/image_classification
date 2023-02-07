from torch import nn


def le_net_wiki(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Flatten(),

        # DenseBlock
        nn.Linear(3136, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, n_classes),
    )
    return cnn_model


def le_net(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.Tanh(),
        nn.Flatten(),

        # DenseBlock
        nn.Linear(2592, 84),
        nn.Tanh(),
        nn.Linear(84, n_classes),
    )
    return cnn_model  # 64 x 64


def alex_net(n_classes):
    cnn_model = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        nn.BatchNorm2d(96),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=0),

        nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=0),

        nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU(),

        nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(384),
        nn.ReLU(),

        nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=2, padding=0),

        nn.Flatten(),

        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, n_classes)
    )
    return cnn_model


def vgg_16_gab(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 4
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 4
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Flatten(),

        # DenseBlock
        nn.Linear(32768, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, n_classes),
        nn.Softmax(dim=-1)
    )
    return cnn_model


def vgg_16(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 4
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 4
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Flatten(),

        # DenseBlock
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, n_classes),
        nn.Softmax(dim=-1)
    )
    return cnn_model


def default_cnn(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 3
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),

        # DenseBlock
        nn.Linear(2592, 256),
        nn.Sigmoid(),
        nn.Linear(256, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, n_classes),
    )
    return cnn_model


def default_test(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(5),
        nn.LeakyReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # ConvBlock 2
        nn.Conv2d(5, 7, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(7),
        nn.LeakyReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(7, 9, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(9),
        nn.LeakyReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(9, 11, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(11),
        nn.LeakyReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(11, 13, kernel_size=5, stride=1, padding=0),
        nn.BatchNorm2d(13),
        nn.LeakyReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

        # nn.Conv2d(13, 15, kernel_size=3, stride=1, padding=0),
        # nn.BatchNorm2d(15),
        # nn.LeakyReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        # DenseBlock
        nn.Flatten(),
        # nn.Linear(117, 117),
        # nn.LeakyReLU(),
        nn.Linear(325, 20),
        nn.LeakyReLU(),
        nn.Linear(20, n_classes),
    )
    return cnn_model  # 80 x 80


def kaggle_test_2(n_classes):
    cnn_model = nn.Sequential(
        # ConvBlock 1
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(p=0.25, inplace=True),

        # ConvBlock 2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(p=0.25, inplace=True),

        # ConvBlock 3
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(p=0.25, inplace=True),

        # DenseBlock
        nn.Flatten(),
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(inplace=True),
        nn.Linear(512, n_classes),
    )
    return cnn_model  # 128x128


def get_cnn_model_by_name(name='le_net', n_classes=2):
    if name == 'le_net':
        return default_cnn(n_classes)
    elif name == 'le_net_wiki':
        return le_net_wiki(n_classes)
    elif name == 'vgg_16':
        return vgg_16(n_classes)
    elif name == 'vgg_16_gab':
        return vgg_16_gab(n_classes)
    elif name == 'alex_net':
        return alex_net(n_classes)
    elif name == 'default':
        return default_test(n_classes)
    else:
        return kaggle_test_2(n_classes)
