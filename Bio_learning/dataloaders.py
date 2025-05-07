from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



def get_data_cifar():
    train_data = datasets.CIFAR10(
        root='~/data/CIFAR10/train',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.CIFAR10(
        root='~/data/CIFAR10/test',
        train=False,
        transform=ToTensor(),
        download=True,
    )

    loaders = {
        'train': DataLoader(train_data,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=4),

        'test': DataLoader(test_data,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=4),
    }
    return loaders



def get_data(data_name='cifar10'):
    if data_name == 'cifar10':
        return get_data_cifar()
    raise NotImplementedError()