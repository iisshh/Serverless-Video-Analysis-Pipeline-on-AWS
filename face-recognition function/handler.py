# Copyright 2024, VISA Lab
# Licensed under the MIT License

import os
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import boto3

s3_client = boto3.client('s3')

os.environ['TORCH_HOME'] = '/tmp/'

face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)
face_embedder = InceptionResnetV1(pretrained='vggface2').eval()

def recognize_faces_in_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    detected_boxes, _ = face_detector.detect(image)

    image_name = os.path.splitext(os.path.basename(image_path))[0].split(".")[0]
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    face, probability = face_detector(image, return_prob=True, save_path=None)

    data_bucket = '1225364608-data'
    local_data_path = '/tmp/data.pt'
    s3_client.download_file(data_bucket, 'data.pt', local_data_path)

    saved_data = torch.load('/tmp/data.pt')
    embedding = face_embedder(face.unsqueeze(0)).detach()
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    distances = []
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(embedding, emb_db).item()
        distances.append(dist)
    min_index = distances.index(min(distances))

    with open("/tmp/" + image_name + ".txt", 'w+') as file:
        file.write(name_list[min_index])
    return name_list[min_index]

def handler(event, context):
        bucket_name = event['bucket']
        image_file = event['image']

        image_key = os.path.join(bucket_name, image_file)
        local_image_path = '/tmp/' + image_file

        s3_client.download_file(bucket_name, image_file, local_image_path)

        recognized_name = recognize_faces_in_image(local_image_path)

        result_file = os.path.splitext(image_file)[0] + '.txt'
        local_result_path = '/tmp/' + result_file

        with open(local_result_path, 'w') as file:
            file.write(recognized_name)

        output_bucket = '1225364608-output'

        s3_client.upload_file(local_result_path, output_bucket, result_file)

        return