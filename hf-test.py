from transformers import AutoImageProcessor, MobileNetV1ForImageClassification
from PIL import Image
import requests
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_0.75_192")
# model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_0.75_192")
hf_model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224").eval()
class TorchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = hf_model.mobilenet_v1
        self.dropout = hf_model.dropout
        self.classifier = hf_model.classifier

    def forward(self, x):
        return self.classifier(self.dropout(self.features(x)['pooler_output']))

model = TorchModel()

preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")

preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.PILToTensor(),
        preprocessor,
        lambda x: x['pixel_values'][0],
    ]
)

device = "cuda"
model = model.to(device)

valid_dataset = torchvision.datasets.ImageNet('evaluation/imagenet/data/ImageNet/', split='val', transform=preprocess)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=512, num_workers=4)

matches = []
for imgs, labels in tqdm(valid_loader):
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(imgs)
    matches.append(logits.argmax(-1) == labels+1)

print(torch.cat(matches).float().mean())
