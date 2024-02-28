import boto3  
import dotenv
from dotenv import load_dotenv  
import os  
import json  
import base64  

load_dotenv()
print(os.getenv('AWS_REGION'))


runtime_client = boto3.client('runtime.sagemaker', region_name="us-east-1")  
  
# Replace 'ENDPOINT_NAME' with your endpoint name  
endpoint_name = 'testb'  
# Path to your image file  
file_path = 'test.png'  
  
# Assuming you still have your endpoint_name and file_path  
with open(file_path, "rb") as image_file:  
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
      
# Create a dictionary to contain the image data  
data = json.dumps({'image': encoded_string})  
  
response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,  
                                          ContentType='application/json',  # Notice we now send JSON  
                                          Body=data)  
  
result = json.loads(response['Body'].read().decode())  
print(result)