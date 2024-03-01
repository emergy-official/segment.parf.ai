resource "aws_iam_role_policy" "update_lambda_policy" {
  name = "${var.prefix}-update-lambda-policy"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "UpdateLambda",
        "Effect" : "Allow",
        "Action" : [
          "lambda:UpdateFunctionCode"
        ],
        "Resource" : [
          "${aws_lambda_function.segment_api.arn}"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecr_login_policy" {
  name = "${var.prefix}-ecr-login"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "ERCLogin",
        "Effect" : "Allow",
        "Action" : [
          "ecr:GetAuthorizationToken"
        ],
        "Resource" : "*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "s3_pull" {
  count = terraform.workspace == "dev" ? 1 : 0

  name = "${var.prefix}-s3-pull"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "s3pull",
        "Effect" : "Allow",
        "Action" : [
          "s3:GetObject",
          "s3:ListBucket"
        ],
        "Resource" : [
          aws_s3_bucket.artifacts[count.index].arn,
          "${aws_s3_bucket.artifacts[count.index].arn}/*"
        ]
      }
    ]
  })
}
resource "aws_iam_role_policy" "allow_dynamobb" {
  name = "${var.prefix}-allow-dynamodb"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "allowdynamodb",
        "Effect" : "Allow",
        "Action" : [
          "dynamodb:BatchWriteItem",
          "dynamodb:Query"
        ],
        "Resource" : [
          aws_dynamodb_table.feedback.arn
        ]
      }
    ]
  })
}
resource "aws_iam_role_policy" "send_mail" {
  name = "${var.prefix}-allow-email"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "sendMail",
        "Effect" : "Allow",
        "Action" : [
          "ses:SendEmail",
          "ses:SendRawEmail"
        ],
        "Resource" : [
          "*"
        ]
      }
    ]
  })
}

// https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push.html
resource "aws_iam_role_policy" "ecr_push_policy" {
  name = "${var.prefix}-ecr-push"
  role = var.github_action_role_id
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Sid" : "ECRPush",
        "Effect" : "Allow",
        "Action" : [
          "ecr:CompleteLayerUpload",
          "ecr:GetAuthorizationToken",
          "ecr:UploadLayerPart",
          "ecr:InitiateLayerUpload",
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage"
        ],
        "Resource" : "${aws_ecr_repository.python_scikit_learn.arn}"
      }
    ]
  })
}
