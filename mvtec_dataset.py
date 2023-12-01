import os
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms

# List of objects in the dataset
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

"""
Directory should be organized as follows:

data
    object_name
        train
            good
                image1.png
                image2.png
        test
            good
                image1.png
                image2.png
                ...
            defect1
                image1.png
                image2.png
                ...
            defect2
                image1.png
                image2.png
                ...
            ...
        ground_truth
            defect1
                image1_mask.png
                image2_mask.png
                ...
            defect2
                image1_mask.png
                image2_mask.png
                ...
"""

class MVTecDataset(VisionDataset):
    
    """ 
    MVTec Dataset for training Anomaly Detection.

    Args:
        root (string): Root directory of dataset.
        object_name (string): Name of object in dataset to extract.
        training (bool, optional): Creates dataset from training set if true, else creates from test set.
        transform (callable, optional): A function/transform that takes in an image and transforms it.
        mask_transform (callable, optional): A function/transform that takes in the mask and transforms it.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # paths to train/test images and masks
    mask_dir= "ground_truth"
    test_dir = "test"
    train_dir= "train"

    # size to resize to
    image_size = 256

    def __init__(self, 
                 root, 
                 object_name, 
                 training = True,
                 input_transform = None,
                 mask_transform = None,
                 target_transform = None):

        super(MVTecDataset, self).__init__(root)

        if object_name not in OBJECTS:
            raise ValueError("Object not in dataset")

        self.root = root
        self.object_name = object_name
        self.training = training
        self.transforms = input_transform
        self.mask_transforms = mask_transform
        self.target_transforms = target_transform

        # directory of images based on training flag
        self.data_dir = os.path.join(self.root, self.object_name, self.train_dir if training else self.test_dir)

        # get images, masks, and labels
        self.images, self.masks, self.labels = self.load_dataset_folder()

    def __getitem__(self, index):

        # get image, mask, and label
        img = self.images[index]
        mask = self.masks[index]
        label = self.labels[index]

        # apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        
        if self.target_transforms:
            label = self.target_transforms(label)

        return img, mask, label
        
    def __len__(self):
        return len(self.images)

    def load_image(self, path, mode="RGB"):
        # if path is None, return black image (no anomaly)
        if path ==  None:
            img = Image.new(mode, (self.image_size, self.image_size))
            return transforms.ToTensor()(img)

        # otherwise, load image and resize
        img = Image.open(path).convert(mode)
        img = img.resize((self.image_size, self.image_size))

        img = transforms.ToTensor()(img)

        return img

    def construct_mask_path(self, img_pth):
        # construct mask path based on path of test image
        dir_change = img_pth.replace(self.test_dir, self.mask_dir)
        mask_path = dir_change.replace(".png", "_mask.png")
        return mask_path

    def load_dataset_folder(self):

        images = []
        masks = []
        labels = []

        self.classes =  [d.name for d in os.scandir(self.data_dir) if d.is_dir()]

        # iterate through each possible class in directory
        for c in self.classes:
            img_dir = os.path.join(self.data_dir, c)

            # find all png files in directory
            for f in os.scandir(img_dir):
                if f.is_file():
                    if f.name.endswith(".png"):

                        # define image, mask, and label
                        img_pth = os.path.join(img_dir, f.name)
                        mask_pth = None if c == "good" else self.construct_mask_path(img_pth)
                        label = 0 if c == "good" else 1

                        # add to lists
                        images.append(self.load_image(img_pth)) 
                        masks.append(self.load_image(mask_pth,"L"))
                        labels.append(label)

        return images, masks, labels


if __name__ == '__main__': 
    """
    # Testing code
    dataset = MVTecDataset("data", object_name="pill", training=True)
    print(dataset.classes)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    print("Length of dataloader:", len(dataloader))

    for i, data in enumerate(dataloader):
        image, mask, label = data
        print(image.shape)
        print(mask.shape)
        print(label)
    """