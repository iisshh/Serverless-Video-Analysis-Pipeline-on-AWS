# Serverless-Video-Analysis-Pipeline-on-AWS

This project focuses on building a scalable and cost-effective serverless application for video analysis using AWS Lambda and supporting AWS services.

## Project Overview

The application processes videos uploaded by users, extracting frames and performing face recognition. This pipeline is built using AWS Lambda, which handles scaling automatically, allowing the system to efficiently manage varying loads.

### Architecture Diagram

<img width="901" alt="image" src="https://github.com/iisshh/Serverless-Video-Analysis-Pipeline-on-AWS/assets/16882901/8212aefa-c039-47a3-b802-0992047e824a">



*Figure 1: Architecture of the Serverless Video Analysis Pipeline*

## Features

- **Automatic Video Processing**: Videos uploaded to S3 trigger Lambda functions that process the video into frames and perform face recognition.
- **Scalability**: Utilizes AWS Lambda for on-demand scalability without the need to manage server resources.
- **Cost-Effective**: Only incurs costs when processing videos, due to the serverless nature of AWS Lambda.

## Technology Stack

- AWS Lambda
- AWS S3
- AWS SQS
- FFmpeg for video processing
- Pre-trained CNN models (e.g., ResNet-34) for face recognition

## Setup Instructions

### Prerequisites

- AWS account
- Configured AWS CLI
- Basic knowledge of AWS Lambda and S3

### Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iisshh/Serverless-Video-Analysis-Pipeline-on-AWS.git
   cd Serverless-Video-Analysis-Pipeline-on-AWS

2. **Set up AWS resources:**
Create S3 buckets for input and output as specified.
Set up Lambda functions with the provided code.

3. **Configure triggers:**
Configure S3 to trigger the Lambda functions upon new video uploads.

## Usage
Upload videos to the designated S3 bucket, and the system will automatically process and store the results in the output bucket.

## Testing
Run the provided workload generator to simulate video uploads and observe the system's response and scalability.
