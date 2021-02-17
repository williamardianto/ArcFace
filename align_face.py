import glob
import os

from models.mtcnn import MTCNN
from PIL import Image
from utils import face_aligner
import numpy as np

# def align_face(raw_dir, aligned_dir):
#     mtcnn = MTCNN()
#
#     for file in glob.glob(raw_dir):
#         file_splits = file.split(os.path.sep)
#         class_name = file_splits[-2]
#         file_name = file_splits[-1]
#
#         class_dir = os.path.join(aligned_dir, class_name)
#         save_path = os.path.join(class_dir, file_name)
#
#         if not os.path.exists(class_dir):
#             os.makedirs(class_dir)
#
#         face_image = Image.open(file)
#         face_image = np.asarray(face_image)
#
#         try:
#             print('extracting:', file)
#             boxes, _, points = mtcnn.detect(face_image, landmarks=True)
#
#             point = points[0]
#             aligned_face = face_aligner(face_image, landmark=point, image_size='112,112')
#             aligned_face = Image.fromarray(aligned_face)
#             aligned_face.save(save_path)
#
#         except:
#             print('failed to extract:', file)


# mtcnn = MTCNN()
# aligned_dir = './data/aligned/CASIA_above_40'
#
# for file in glob.glob("./data/raw/CASIA/*/*.jpg"):
#     file_splits = file.split(os.path.sep)
#     class_name = file_splits[-2]
#     file_name = file_splits[-1]
#
#     class_dir = os.path.join(aligned_dir, class_name)
#     save_path = os.path.join(class_dir, file_name)
#
#     if not os.path.exists(class_dir):
#         os.makedirs(class_dir)
#
#     face_image = Image.open(file)
#     face_image = np.asarray(face_image)
#
#     try:
#         print('extracting:', file)
#         boxes, _, points = mtcnn.detect(face_image, landmarks=True)
#
#         point = points[0]
#         aligned_face = face_aligner(face_image, landmark=point, image_size='112,112')
#         aligned_face = Image.fromarray(aligned_face)
#         aligned_face.save(save_path)
#
#     except:
#         print('failed to extract:', file)

if __name__ == '__main__':
    mtcnn = MTCNN(device='cuda')
    aligned_dir = './data/aligned/CASIA-maxpy-clean'

    for file in glob.glob("./data/CASIA-maxpy-clean/CASIA-maxpy-clean/*/*.jpg"):
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


    # aligned_dir = './data/aligned/lfw'
    # raw_dir = './data/lfw/*/*.jpg'


    # align_face(raw_dir, aligned_dir)


    # mtcnn = MTCNN()
    # aligned_dir = './data/aligned/CASIA_above_40'
    #
    # for root, dirs, files in os.walk("./data/aligned/CASIA_front"):
    #     if len(files) >= 40:
    #         path_splits = root.split(os.path.sep)
    #         class_name = path_splits[-1]
    #
    #         class_dir = os.path.join(aligned_dir, class_name)
    #         if not os.path.exists(class_dir):
    #             os.makedirs(class_dir)
    #
    #         for f in files:
    #             file_path = os.path.join(root, f)
    #
    #             try:
    #                 face_image = Image.open(file_path)
    #                 # face_image = np.asarray(face_image)
    #
    #                 print('extracting:', file_path)
    #                 # boxes, _, points = mtcnn.detect(face_image, landmarks=True)
    #                 #
    #                 # point = points[0]
    #                 # aligned_face = face_aligner(face_image, landmark=point, image_size='112,112')
    #                 # aligned_face = Image.fromarray(aligned_face)
    #                 face_image.save(os.path.join(class_dir, f))
    #
    #             except:
    #                 print('failed to extract:', file_path)








