import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision import transforms
from Model import Model
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

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

    batch_size = 16
    train_size = dataset_train.number_of_images // batch_size + 1
    val_size = dataset_val.number_of_images // batch_size + 1
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=False)

    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                  num_workers=8, pin_memory=True, drop_last=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #initialize model
    model = Model(channels=3).to(device)

    #create loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, T_mult=1, eta_min=0, last_epoch=-1, verbose=True)

    #schedulers learning rate policies
    epochs = 150
    writer = SummaryWriter(log_dir="/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/logs"
                           , flush_secs=1)
    for e in range(epochs):
        model.train()
        # train loop
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        for i_batch, (images, labels) in enumerate(train_dataloader):
            #print(i_batch, images.shape, labels.shape)
            logits = model(images.to(device))
            softmax = F.softmax(logits, dim=1)

            predicted_labels = torch.argmax(softmax, dim=1).cpu().detach().numpy()

            epoch_train_accuracy += accuracy_score(labels.numpy(), predicted_labels)

            optimizer.zero_grad()
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), e*train_size + i_batch)
            writer.add_scalar('Accuracy/train', accuracy_score(labels.numpy(), predicted_labels), e*train_size + i_batch)

        # # #validation loop
        epoch_val_loss = 0
        epoch_val_accuracy = 0
        # model.eval()
        with torch.no_grad():
            for i_batch, (images, labels) in enumerate(val_dataloader):
                # print(i_batch, images.shape, labels.shape)
                logits = model(images.to(device))
                softmax = F.softmax(logits, dim=1)

                predicted_labels = torch.argmax(softmax, dim=1).cpu().detach().numpy()

                epoch_val_accuracy += accuracy_score(labels.numpy(), predicted_labels)

                loss = criterion(logits, labels.to(device))

                epoch_val_loss += loss.item()

                writer.add_scalar('Loss/val', loss.item(), e * val_size + i_batch)
                writer.add_scalar('Accuracy/val', accuracy_score(labels.numpy(), predicted_labels),
                                  e * val_size + i_batch)

        scheduler.step(epoch_val_loss/val_size)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], e)
        print("epoch {}/{} - train_loss {} - train_acc {} - val_loss {} - val_acc {}".format(e+1, epochs, epoch_train_loss/train_size, epoch_train_accuracy/train_size, epoch_val_loss/val_size, epoch_val_accuracy/val_size))

    #TODO should probably save the best model as regards the minimum loss on val set or save model every N epochs
    checkpoint_file = 'ckpt.pth'
    torch.save(model.state_dict(), "/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/logs/" + checkpoint_file)
