# Create Lambda function to perform CRUD operations on DynamoDB table    
resource "aws_lambda_function" "segment_api" {
  function_name = "${var.prefix}_api"
  package_type  = "Image"
  role          = aws_iam_role.lambda_exec_segment_api.arn
  image_uri     = "${aws_ecr_repository.python_scikit_learn.repository_url}:latest"

  timeout     = 120
  memory_size = 1024
  environment {
    variables = {
    }
  }
}

resource "aws_cloudwatch_log_group" "segment_api" {
  name              = "/aws/lambda/${aws_lambda_function.segment_api.function_name}"
  retention_in_days = 3
}

# IAM role for the Lambda function to access necessary resources  
resource "aws_iam_role" "lambda_exec_segment_api" {
  name = "${var.prefix}_lambda_exec_segment_api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_exec_policy_segment_api" {
  name = "${var.prefix}_lambda_exec_policy_segment_api"
  role = aws_iam_role.lambda_exec_segment_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      },
    ]
  })
}

resource "aws_lambda_permission" "api_gateway_segment" {
  statement_id  = "AllowAPIGatewayInvokeSegment"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.segment_api.arn
  principal     = "apigateway.amazonaws.com"

  # If you have set up an AWS Account Alias, use this line instead  
  # source_arn = "arn:aws:execute-api:${var.region_name}:${var.aws_account_id}:${aws_api_gateway_rest_api.example.id}/${aws_api_gateway_deployment.example.stage_name}/ANY/RESOURCE_PATH"  

  source_arn = "arn:aws:execute-api:${var.region_name}:${var.aws_account_id}:${aws_api_gateway_rest_api.website.id}/*/*/*"
}
