import boto3  
import dotenv
from dotenv import load_dotenv  
import os  
import json  
load_dotenv()
print(os.getenv('AWS_REGION'))


runtime_client = boto3.client('runtime.sagemaker', region_name="us-east-1")  
  
# Replace 'ENDPOINT_NAME' with your endpoint name  
endpoint_name = 'test3'  
data = json.dumps({"data": [[6.5, 3.2, 5.1, 2]]})  # Example input  
  
response = runtime_client.invoke_endpoint(EndpointName=endpoint_name,  
                                          ContentType='application/json',  
                                          Body=data)  
  
result = json.loads(response['Body'].read().decode())  
print(result)  