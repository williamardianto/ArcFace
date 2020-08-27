import glob
import os

from models.mtcnn import MTCNN
from PIL import Image
from utils import face_aligner
import numpy as np

if __name__ == '__main__':

    mtcnn = MTCNN()
    aligned_dir = './data/aligned/CASIA'

    for file in glob.glob("./data/raw/CASIA/*/*.jpg"):
        file_splits = file.split(os.path.sep)
        class_name = file_splits[-2]
        file_name = file_splits[-1]

        class_dir = os.path.join(aligned_dir, class_name)
        save_path = os.path.join(class_dir, file_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        face_image = Image.open(file)
        face_image = np.asarray(face_image)

        try:
            print('extracting:', file)
            boxes, _, points = mtcnn.detect(face_image, landmarks=True)

            point = points[0]
            aligned_face = face_aligner(face_image, landmark=point, image_size='112,112')
            aligned_face = Image.fromarray(aligned_face)
            aligned_face.save(save_path)

        except:
            print('failed to extract:', file)


