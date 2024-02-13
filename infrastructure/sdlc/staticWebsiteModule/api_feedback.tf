# Create Lambda function to perform CRUD operations on DynamoDB table    
resource "aws_lambda_function" "feedback_api" {
  function_name = "${var.prefix}_feedback_api"
  handler       = "index.handler"
  role          = aws_iam_role.lambda_exec_feedback_api.arn
  runtime       = "nodejs20.x"
  filename      = data.archive_file.lambda_feedback_zip.output_path
  # Do not update the lambda, it will be done by Github CI/CD
  source_code_hash = data.archive_file.lambda_feedback_zip.output_base64sha256
  layers           = [aws_lambda_layer_version.lambda_feedback_api_lambda_layer.arn]

  timeout     = 10
  memory_size = 128
  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.feedback.name
    }
  }
}

resource "aws_cloudwatch_log_group" "feedback_api" {
  name              = "/aws/lambda/${aws_lambda_function.feedback_api.function_name}"
  retention_in_days = 3
}

# resource "aws_lambda_function_url" "feedback_api" {
#   function_name      = aws_lambda_function.feedback_api.function_name
#   authorization_type = "NONE"

#   cors {
#     allow_origins  = ["*"]
#     allow_methods  = ["*"]
#     allow_headers  = ["*"]
#     expose_headers = ["*"]
#     max_age        = 0
#   }
# }

# IAM role for the Lambda function to access necessary resources  
resource "aws_iam_role" "lambda_exec_feedback_api" {
  name = "${var.prefix}_lambda_exec_feedback_api"

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

resource "aws_iam_role_policy" "lambda_exec_policy_feedback_api" {
  name = "${var.prefix}_lambda_exec_policy_feedback_api"
  role = aws_iam_role.lambda_exec_feedback_api.id

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

resource "aws_iam_role_policy" "lambda_exec_policy_feedback_api_db" {
  name = "${var.prefix}_lambda_exec_policy_feedback_api_db"
  role = aws_iam_role.lambda_exec_feedback_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "dynamodb:BatchWriteItem",
          "dynamodb:Query",
        ]
        Effect   = "Allow"
        Resource = aws_dynamodb_table.feedback.arn
      },
    ]
  })
}

resource "aws_iam_role_policy" "lambda_exec_policy_feedback_api_metrics" {
  name = "${var.prefix}_lambda_exec_policy_feedback_api_metrics"
  role = aws_iam_role.lambda_exec_feedback_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}

resource "aws_lambda_permission" "api_gateway_feedback" {
  statement_id  = "AllowAPIGatewayInvokeFeedback"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.feedback_api.arn
  principal     = "apigateway.amazonaws.com"

  # If you have set up an AWS Account Alias, use this line instead  
  # source_arn = "arn:aws:execute-api:${var.region_name}:${var.aws_account_id}:${aws_api_gateway_rest_api.example.id}/${aws_api_gateway_deployment.example.stage_name}/ANY/RESOURCE_PATH"  

  source_arn = "arn:aws:execute-api:${var.region_name}:${var.aws_account_id}:${aws_api_gateway_rest_api.website.id}/*/*/*"
}

# Code of the lambda functions
data "archive_file" "lambda_feedback_zip" {
  type        = "zip"
  source_dir  = "${var.api_path}/dist/feedback-lambda"
  output_path = "${var.api_path}/dist/feedback-lambda.zip"
}


# Code of the lambda layer
data "archive_file" "lambda_feedback_api_lambda_layer" {
  type        = "zip"
  source_dir  = "${var.api_path}/dist/nodejs"
  output_path = "${var.api_path}/dist/nodejs.zip"
}

# Create the Lambda layer  
resource "aws_lambda_layer_version" "lambda_feedback_api_lambda_layer" {
  layer_name          = "${var.prefix}_feedback_lambda_layer"
  filename            = data.archive_file.lambda_feedback_api_lambda_layer.output_path
  source_code_hash    = filebase64sha256(data.archive_file.lambda_feedback_api_lambda_layer.output_path)
  compatible_runtimes = ["nodejs20.x"]
}

resource "aws_sns_topic" "negative_feedback_alarm_topic" {  
  name = "${var.prefix}-negative-feedback-alarm-topic"  
}

# Subscribe the Lambda function to the SNS topic  
resource "aws_sns_topic_subscription" "lambda_subscription" {
  topic_arn = aws_sns_topic.negative_feedback_alarm_topic.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.email_alert.arn
}

resource "aws_cloudwatch_metric_alarm" "negative_feedback_alarm" {
  alarm_name          = "NegativeFeedbackAlarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "1"
  metric_name         = "Sentiment"
  namespace           = "segment.parf.ai"
  period              = "300"
  statistic           = "Sum"
  threshold           = "3"

  dimensions = {
    "Feedback" = "Negative"
  }

  actions_enabled   = true
  alarm_description = "This alarm fires when there are 3 negatives ƒfeedback within 5 minutes"

  // Attach the proper SNS topic ARN or other actions you want to trigger  
  alarm_actions = [aws_sns_topic.negative_feedback_alarm_topic.arn]

  // Use insufficient_data_actions if you want to trigger an action when the data is missing  
  // Use ok_actions to trigger an action when the metric returns to a normal state  
}

# Allow the SNS topic to invoke the Lambda function  
resource "aws_lambda_permission" "sns_invoke_email_alert_lambda" {  
  statement_id  = "AllowExecutionFromSNS"  
  action        = "lambda:InvokeFunction"  
  function_name = aws_lambda_function.email_alert.arn  
  principal     = "sns.amazonaws.com"  
  source_arn    = aws_sns_topic.negative_feedback_alarm_topic.arn  
}  
  