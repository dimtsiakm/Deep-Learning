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
    PATH = '/media/dimitris/data_linux/Deep Learning Assignment/CNN/dataset_human_position/logs/cosine_annealing/ckpt.pth'
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
        rgb = np.expand_dims(rgb, axis=0)
        image = data_transforms(rgb).to(device)


        with torch.no_grad():
            out = model(image)
            softmax = F.softmax(out, dim=1)
            predicted_label = torch.argmax(softmax, dim=1).cpu().detach().numpy()
            predicted_label = get_key(predicted_label[0])

        cv2.putText(frame, predicted_label, (frame.size[0]//2, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()