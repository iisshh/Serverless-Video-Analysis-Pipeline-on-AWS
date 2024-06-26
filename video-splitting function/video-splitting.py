import os
import subprocess
import math
import boto3
import json

s3_client = boto3.client('s3')

def split_video_into_frames(video_file):
    filename = os.path.basename(video_file)
    file_identifier = os.path.splitext(filename)[0]
    output_directory = os.path.join("/tmp", file_identifier)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_image_path = os.path.join(output_directory, f"{file_identifier}.jpg")

    split_command = f'/opt/bin/ffmpeg -ss 0 -r 1 -i {video_file} -vf fps=1/10 -start_number 0 -vframes 1 {output_image_path} -y'
    
    try:
        subprocess.check_call(split_command, shell=True)
    except subprocess.CalledProcessError as ex:
        print(ex.returncode)
        print(ex.output)

    fps_command = f'ffmpeg -i {video_file} 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"'
    fps_output = subprocess.check_output(fps_command, shell=True).decode("utf-8").rstrip("\n")
    frames_per_second = math.ceil(float(fps_output))
    
    return output_directory

def handler(event_data, runtime_context):
    source_bucket = event_data['Records'][0]['s3']['bucket']['name']
    source_key = event_data['Records'][0]['s3']['object']['key']

    video_file_path = os.path.join('/tmp', source_key)
    s3_client.download_file(source_bucket, source_key, video_file_path)

    frames_directory = split_video_into_frames(video_file_path)
    frames_folder_name = frames_directory.split('/')[-1]

    for root, dirs, files in os.walk(frames_directory):
        for file in files:
            frame_path = os.path.join(root, file)
            s3_client.upload_file(frame_path, "1225364608-stage-1", file)

    lambda_client = boto3.client('lambda')

    target_function_name = 'face-recognition'
    payload = {"bucket": "1225364608-stage-1", "image": file}

    lambda_client.invoke(
        FunctionName=target_function_name,
        InvocationType='Event',
        Payload=json.dumps(payload)
    )
