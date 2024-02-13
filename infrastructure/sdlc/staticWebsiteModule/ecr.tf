resource "aws_ecr_repository" "python_scikit_learn" {
  name = "${var.prefix}-python-scikit-learn"
}