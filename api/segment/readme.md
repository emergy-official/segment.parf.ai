# Segment.parf.ai - Segment API

## Introduction

The segment API returns the segment prediction of a text from a trained model.  
It loads the tokenizer & model that were previously trained and predicts a given text.

This API is executed on AWS Lambda through a Docker container.  
Using free tier, the first execution will take ~30sec-1min time to start and will then stop after ~15min of inactiivty. (reserved concurrency lambda makes it always available for some $$)

The response time should be ~300ms per request after coldstart.

## API Usage example

- Prod: https://segment.parf.ai/api
- Dev: https://dev.segment.parf.ai/api

### Get segment response

```js
var request = require("request");
var options = {
  method: "POST",
  url: "https://dev.segment.parf.ai/api/segment",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    text: "I am so happy, this is very good news, congrats!",
  }),
};

request(options, function (error, response) {
  if (error) throw new Error(error);
  console.log(response.body);
});
```

Response example
```json
{
    "segment": 0.9474098682403564
}
```

## How to test locally

In order to test locally you need to have the following structure:

```sh
# within api/segment

- artifacts
  - model.pkl
  - params.json
  - tokenizer.pkl
```

The artifacts are available after each experiments on MLFlow.  
The MLFlow server run locally and push the artifacts to S3 in order to reuse them within Github actions

```sh

# Create the environment
conda create -p venv python=3.11.7 -y

# Use the environment
conda activate venv/

# Install dependencies
pip install -r requirements.txt

# feel free to modify the text
python local_test.py 

# Start unit test, the one used by Github actions
python -m unittest unit_test.py
```

## How to build & push the docker locally

```sh
cd api/segment

# Authentication
export DEV_ACCOUNT_ID=REPLACE_ME
export PROD_ACCOUNT_ID=REPLACE_ME
export INFRA_ACCOUNT_ID=REPLACE_ME
export ACCOUNT_ID=$DEV_ACCOUNT_ID

# Login from infra account to dev account
eval $(aws sts assume-role --profile $INFRA_ACCOUNT_ID --role-arn "arn:aws:iam::"$ACCOUNT_ID":role/provision" --role-session-name AWSCLI-Session | jq -r '.Credentials | "export AWS_ACCESS_KEY_ID=\(.AccessKeyId)\nexport AWS_SECRET_ACCESS_KEY=\(.SecretAccessKey)\nexport AWS_SESSION_TOKEN=\(.SessionToken)\n"')

# Download the artifact
rm -rf artifacts
export FILE_PATH="657967221979013195/2f2f781caad24db9bbc386eef1fbde7b"
aws s3 cp "s3://artifact-dev.segment.parf.ai/$FILE_PATH" ./ --recursive
aws s3 cp artifacts "s3://artifact-dev.segment.parf.ai/657967221979013195/2f2f781caad24db9bbc386eef1fbde7b/artifacts" --recursive

# Login to AWS ECR to push docker
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/

# Tag version image to latest
export IMG_VERSION=latest

# Logout to the public ECR to make sure that the private publish won't fail
docker logout public.ecr.aws

# Build the image
docker build --platform linux/amd64 -t python-scikit-learn:$IMG_VERSION .

# Tag the image so you can push it to AWS ECR (private repo)
docker tag python-scikit-learn:$IMG_VERSION "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/segment-python-scikit-learn:latest

# Push the image
docker push "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/segment-python-scikit-learn:latest

# Update the lambda to use the new image
aws lambda update-function-code --function-name segment_api --image-uri "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/segment-python-scikit-learn:latest --region us-east-1

## OPTIONNAL: RUN THE DOCKER ImageLOCALLY

docker run --platform linux/amd64 -p 9000:8080 python-scikit-learn:$IMG_VERSION

curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"body": "{\"text\":\"I am so happy this is great news, congrats!\"}"}'
```

## Structure

- **Dockerfile**: To build the container
- **lambda_function**: Python script invoked by the API
- **local_test.py**: Python script to test the segment locally
- **requirements.txt**: Libraries used
- **unit_test.py**: Unit test done within Github actions
