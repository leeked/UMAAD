import os
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms

OBJECTS = ["bottle", 
           "cable", 
           "capsule", 
           "carpet", 
           "grid", 
           "hazelnut", 
           "leather", 
           "metal_nut", 
           "pill", 
           "screw", 
           "tile", 
           "toothbrush", 
           "transistor", 
           "wood", 
           "zipper"]

class MVTecDataset(VisionDataset):
    

    # paths to images and masks
    mask_dir= "ground_truth"
    test_dir = "test"
    train_dir= "train"

    # image size
    image_size = 256

    def __init__(self, root, object_name, training):
        super(MVTecDataset, self).__init__(root)

        if object_name not in OBJECTS:
            raise ValueError("Object not in dataset")

        self.root = root
        self.object_name = object_name
        self.training = training

        self.data_dir = os.path.join(self.root, self.object_name, self.train_dir if training else self.test_dir)
        self.classes, self.class_to_label = self.get_classes(self.data_dir)
        self.images, self.masks, self.labels = self.load_dataset_folder()
        
        # transforms
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

        self.mask_transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):

        img = self.images[index]
        mask = self.masks[index]
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(img)
        
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        return img, mask, label
        
    def __len__(self):
        return len(self.images)

    def get_classes(self, directory):

        classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name == "good"]
        classes.insert(0, "good") # insert good to beginning of array (0 index for no anomaly)
    
        class_to_label = {c: i for i, c in enumerate(classes)}

        return classes, class_to_label

    def load_image(self, path, mode="RGB"):

        if path ==  None:
            return Image.new(mode, (self.image_size, self.image_size))

        img = Image.open(path).convert(mode)
        img = img.resize((self.image_size, self.image_size))

        return img

    def construct_mask_path(self, img_pth):
        dir_change = img_pth.replace(self.test_dir, self.mask_dir)
        mask_path = dir_change.replace(".png", "_mask.png")
        return mask_path

    def load_dataset_folder(self):

        images = []
        masks = []
        labels = []

        for c in self.classes:
            img_dir = os.path.join(self.data_dir, c)
            for f in os.scandir(img_dir):
                if f.is_file():
                    if f.name.endswith(".png"):

                        img_pth = os.path.join(img_dir, f.name)
                        mask_pth = None if c == "good" else self.construct_mask_path(img_pth)
                        label = 0 if c == "good" else self.class_to_label[c]

                        images.append(self.load_image(img_pth)) 
                        masks.append(self.load_image(mask_pth,"L"))
                        labels.append(label)

        return images, masks, labels


if __name__ == '__main__': 
    # Testing code
    dataset = MVTecDataset("data", object_name="screw", training=False)
    print(dataset.classes)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    print("Length of dataloader:", len(dataloader))

    for i, data in enumerate(dataloader):
        image, mask, label = data
        print(image.shape)
        print(mask.shape)
        print(label)

    #cd documents/github/super-duper-deep-learning