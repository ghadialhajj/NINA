import torch
from architechture import UNet
from dataset import Land
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.nn import CrossEntropyLoss
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

training_dataset = Land(root_dir='/home/ghadi/PycharmProjects/test-multiclass-segmentation/dset-s2-grunnkart',
                        images_path='tra_scene', labels_path='tra_truth')
training_data_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)

validation_dataset = Land(root_dir='/home/ghadi/PycharmProjects/test-multiclass-segmentation/dset-s2-grunnkart',
                          images_path='val_scene', labels_path='val_truth')
validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

loss_fn = CrossEntropyLoss()

encoder = resnet18

model = UNet(encoder, pretrained=True, out_channels=6)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def jaccard(output, target):
    output = output.argmax(dim=1)
    output = output.view(-1)
    target = target.view(-1)
    per_class = jaccard_score(output, target, average=None)
    avg = per_class.mean()
    return per_class, avg


# training loop
for epoch in tqdm(range(40), desc="Epochs"):
    model.train()
    for data, labels in tqdm(training_data_loader, desc=f"Training {epoch + 1}", leave=False):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    for data, labels in tqdm(validation_data_loader, desc=f"Evaluation {epoch + 1}", leave=False):
        with torch.no_grad():
            output = model(data)
            loss = loss_fn(output, labels)
            per_class, avg = jaccard(output, labels)
            print(f'Epoch: {epoch}, Loss: {loss}, Average Jaccard Score: {avg}, Per Class Jaccard Score: {per_class}')

# plot the results

data, labels = training_dataset[0]
output = model(data.unsqueeze(0))
output = output.argmax(dim=1)
_, axs = plt.subplots(1, 3, figsize=(15, 6))

# Define the class labels and corresponding colors
class_labels = ['urban', 'cropland', 'grass', 'forest', 'wetland', 'water']
class_colors = ['red', 'yellow', 'lime', 'green', 'purple', 'blue']

rgb = data.permute(1, 2, 0).numpy()
axs[0].imshow(rgb.clip(min=0, max=1))
axs[0].set_title("RGB Image")

axs[1].imshow(labels.data.squeeze(), cmap=ListedColormap(class_colors))
axs[1].set_title("Class Mask")

axs[2].imshow(output.data.squeeze(), cmap=ListedColormap(class_colors))
axs[2].set_title("Prediction")

legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
                   for color, label in zip(class_colors, class_labels)]

axs[1].legend(handles=legend_elements, loc='upper right', title="Classes")

plt.tight_layout()
plt.show()
