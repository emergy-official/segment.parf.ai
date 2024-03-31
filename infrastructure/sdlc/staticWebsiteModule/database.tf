
resource "aws_dynamodb_table" "feedback" {
  name           = "${var.prefix}-feedback"
  billing_mode   = "PROVISIONED"
  read_capacity  = 1
  write_capacity = 1
  hash_key       = "partition"
  range_key      = "timestamp"
  
  attribute {
    name = "partition"
    type = "N"
  }

  attribute {
    name = "timestamp"
    type = "N"
  }
}
