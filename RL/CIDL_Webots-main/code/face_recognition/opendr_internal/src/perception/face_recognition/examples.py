from perception.face_recognition.face_recognition_learner import FaceRecognitionLearner
from engine.datasets import ExternalDataset
import os
import numpy as np


def test_fit(self):
    recognizer = FaceRecognitionLearner(backbone='mobilefacenet', mode='full', device="cpu",
                                        temp_path=self.temp_dir, iters=2,
                                        batch_size=2, checkpoint_after_iter=0)
    dataset_path = os.path.join(self.temp_dir, 'test_data/images')
    train = ExternalDataset(path=dataset_path, dataset_type='imagefolder')
    results = recognizer.fit(dataset=train, silent=False, verbose=True)
    self.assertNotEqual(len(results), 0)


def test_align(self):
    imgs = os.path.join(self.temp_dir, 'test_data/images')
    self.recognizer.load(self.temp_dir)
    self.recognizer.align(imgs, os.path.join(self.temp_dir, 'aligned'))
    self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'aligned')))


def test_fit_reference(self):
    imgs = os.path.join(self.temp_dir, 'test_data/images')
    save_path = os.path.join(self.temp_dir, 'reference')
    self.recognizer.load(self.temp_dir)
    self.recognizer.fit_reference(imgs, save_path)


def test_infer(self):
    imgs = os.path.join(self.temp_dir, 'test_data/images')
    save_path = os.path.join(self.temp_dir, 'reference')
    self.recognizer.load(self.temp_dir)
    self.recognizer.fit_reference(imgs, save_path)
    img = np.random.random((112, 112, 3))
    result = self.recognizer.infer(img)


def test_eval(self):
    self.recognizer.load(self.temp_dir)
    dataset_path = os.path.join(self.temp_dir, 'test_data/images')
    eval_dataset = ExternalDataset(path=dataset_path, dataset_type='imagefolder')
    results = self.recognizer.eval(eval_dataset, num_pairs=10000)


def test_save_load(self):
    save_path = os.path.join(self.temp_dir, 'saved')
    self.recognizer.backbone_model = None
    self.recognizer.load(self.temp_dir)
    self.assertIsNotNone(self.recognizer.backbone_model, "model is None after loading pth model.")
    self.recognizer.save(save_path)
    self.assertTrue(os.path.exists(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.pth')))
    self.assertTrue(os.path.exists(os.path.join(save_path, 'backbone_' + self.recognizer.backbone + '.json')))


def test_optimize(self):
    self.recognizer.load(self.temp_dir)
    self.recognizer.optimize()


def test_download(self):
    download_path = os.path.join(self.temp_dir, 'downloaded')
    check_path = os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.pth')
    check_path_json = os.path.join(download_path, 'backbone_' + self.recognizer.backbone + '.json')
    self.recognizer.download(download_path, mode="pretrained")

