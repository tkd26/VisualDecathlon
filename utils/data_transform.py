from torchvision.transforms import transforms
import pickle

# define data transformation
def data_transform(data_path, name, train=True):
    with open(data_path + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(72),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(72),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test