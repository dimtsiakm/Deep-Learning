import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision import transforms


class HumanDataset(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir

        mode_list = ['train', 'test', 'val']
        if mode in mode_list:
            self.mode = mode
        else:
            raise ValueError('Mode should be train, test, val')

        self.data_dir = ''
        self.dataset_classes = []
        self.number_of_images = 0
        self.labels = {'right': 0, 'center': 1, 'left': 2, 'noposition': 3}

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            #TODO: revise transformation for validation and test (TenCrop).
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        self.images = []

        self.parseData()
        self.create_list()

    def parseData(self):
        self.data_dir = os.path.join(self.root_dir, self.mode)
        self.dataset_classes = os.listdir(self.data_dir)
        for i, cl in enumerate(self.dataset_classes):
            self.number_of_images += len(os.listdir(os.path.join(self.data_dir, cl)))

    def create_list(self):
        for cl in self.dataset_classes:
            current_list = os.listdir(os.path.join(self.data_dir, cl))

            for img in current_list:
                img_path = os.path.join(self.data_dir, cl, img)
                self.images.append([img_path, self.labels[cl]])

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.data_transforms[self.mode](image)
        return image, self.images[idx][1]

if __name__ == '__main__':
    dataset_train = HumanDataset('/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/data', 'train')
    dataset_val = HumanDataset('/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/data', 'val')

    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=False)

    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=False)

    #initialize model
    #create loss criterion
    #schedulers learnig rate policies
    epochs = 20
    for e in range(epochs):
        # train loop
        for i_batch, (images, labels) in enumerate(train_dataloader):
            print(i_batch, images.shape, labels.shape)
            #set model to train mode
            #feed input to model -> forward pass
            #get batch results -> print/visualize/report
            #backward pass
            #optimizer call


        # #validation loop
        for i_batch, (images, labels) in enumerate(val_dataloader):
            print(i_batch, images.shape, labels.shape)
            #set model to eval mode
            #feed input to model -> forward pass
            #get batch results -> print/visualize/report

        #save checkpoint every N epochs

        exit()