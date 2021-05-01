import cv2
from torchvision import transforms
from Model import Model
import torch
import torch.nn.functional as F
import numpy as np

labels = {'right': 0, 'center': 1, 'left': 2, 'noposition': 3}
def get_key(val):
    for key, value in labels.items():
        if val == value:
            return key

    return "key doesn't exist"

if __name__ == '__main__':

    data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(channels=3).to(device)
    PATH = 'D:\Development\Testing\Deep-Learning\Models\ckpt.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = data_transforms(rgb).to(device)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            out = model(image)
            softmax = F.softmax(out)
            predicted_label = torch.argmax(softmax).cpu().detach().numpy()
            predicted_label = get_key(predicted_label)

        cv2.putText(frame, predicted_label, (256, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()