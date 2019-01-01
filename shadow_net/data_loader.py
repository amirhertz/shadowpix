import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if not '.jpg' in fname:
                    continue

                path = os.path.join(root, fname)
                item = (path)
                images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.resize((64, 64)).convert('L')


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, root, k=4, transform=ToTensor()):
        self.k = k

        samples = make_dataset(root)

        self.root = root
        self.loader = pil_loader

        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        samples = []
        for i in range(self.k):
            path = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            samples.append(sample)

        return torch.stack(samples, 0)

    def __len__(self):
        return len(self.samples) // self.k

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
