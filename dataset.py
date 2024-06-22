from torch.utils.data import Dataset, DataLoader
import os
import rioxarray as rxr
import torch


class Land(Dataset):
    def __init__(self, root_dir, images_path, labels_path, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        self.images_path = images_path
        self.labels_path = labels_path

        self.images = os.listdir(os.path.join(root_dir, images_path))
        self.labels = os.listdir(os.path.join(root_dir, labels_path))

        self.images.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = rxr.open_rasterio(os.path.join(self.root_dir, self.images_path, self.images[idx]))
        labels = rxr.open_rasterio(os.path.join(self.root_dir, self.labels_path, self.labels[idx]))

        data = data.data[[2, 1, 0]] / 2000
        labels = labels.data[0] - 1

        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()

        # crop the image to 840, 840
        data = data[..., 0:512, 0:512]
        labels = labels[0:512, 0:512]

        if self.transform:
            data = self.transform(data)
            labels = self.transform(labels)

        return data, labels


if __name__ == '__main__':
    dataset = Land(root_dir='/home/ghadi/PycharmProjects/test-multiclass-segmentation/dset-s2-grunnkart',
                   images_path='tra_scene', labels_path='tra_truth')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data, labels in data_loader:
        print(data, labels)
        break
