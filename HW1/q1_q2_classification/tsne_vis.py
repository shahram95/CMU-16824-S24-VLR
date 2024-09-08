import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from voc_dataset import VOCDataset
from torch.utils.data import DataLoader

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ResNet model without pre-trained weights
model = models.resnet18()
model.load_state_dict(torch.load('model.pth'))

model.fc = nn.Identity()
model = model.to(device)
model.eval()

# Prepare the dataset
dataset = VOCDataset(split='val', size=224)
indices = np.random.choice(len(dataset), size=1000, replace=False)

# Extract features and labels
features = []
labels = []

for i in indices:
    img, label, _ = dataset[i]
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = model(img)
    features.append(feature.squeeze().cpu().numpy())
    labels.append(label)

features = np.array(features)
labels = [label.argmax().item() for label in labels]
labels = np.array(labels)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=100, learning_rate=1000, n_iter=10000, random_state=0)
projections = tsne.fit_transform(features)

# Visualization
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
legend_elements = []

for i, label in enumerate(unique_labels):
    idxs = (labels == label)
    plt.scatter(projections[idxs, 0], projections[idxs, 1], label=class_names[label], alpha=0.5)
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(2*i % 20), markersize=10, label=class_names[label]))

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
plt.title('t-SNE visualization of Custom Trained ResNet Features')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
