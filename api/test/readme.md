# Tag the image so you can push it to AWS ECR (private repo)
docker tag mltest:latest "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/mltest:latest

# Push the image
docker push "$ACCOUNT_ID".dkr.ecr.us-east-1.amazonaws.com/mltest:latest