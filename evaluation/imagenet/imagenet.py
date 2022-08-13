import torchvision
import os



class ImageNetTest:
    def __init__(self, root: str, transform = None):
        self.data_paths = []
        self.labels = []
        self.transform = transform

        for root, dirs, files in os.walk(root):
            for image_path in files:
                self.data_paths.append(root + '/' + image_path)
                self.labels.append(int(image_path[16:-5]) // 1000)

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.data_paths[index]) / 255
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.labels[index])

    def __len__(self):
        return len(self.data_paths)