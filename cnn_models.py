from torch import nn

# PERDA DE RESOLUÇÃO POR CAMADA: ((IMG_SIZE - KERNEL_SIZE + 2*PADDING)/STRIDE)+1


def le_net(n_classes, image_size):
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
        nn.Linear(103968, 84),
        nn.Tanh(),
        nn.Linear(84, n_classes),
    )
    return cnn_model


def vgg_16(n_classes, image_size):
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


def get_cnn_model_by_name(name='le_net', n_classes=2, image_size=None):
    if name == 'le_net':
        return le_net(n_classes, image_size)
    elif name == 'vgg_16':
        return vgg_16(n_classes, image_size)
    else:
        print(f'{name} not found!')
