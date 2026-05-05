# ============================================================
# iam.tf — IAM Roles & Permissions
# Project: CISC886 Cloud Computing
# NetID: 25nplx
# Purpose: IAM roles for EMR and EC2 to access AWS services
# ============================================================

# EMR Service Role
# Allows the EMR service itself to manage cluster resources
# (launch EC2 nodes, configure networking, etc.)
resource "aws_iam_role" "emr_service_role" {
  name = "25nplx-emr-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Effect = "Allow"
        Principal = {Service = "elasticmapreduce.amazonaws.com"}
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name    = "25nplx-emr-service-role"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Attach AWS managed policy for EMR service
resource "aws_iam_role_policy_attachment" "emr_service_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

# EMR EC2 Instance Profile Role
# This role is attached to the actual EC2 nodes inside EMR
# Allows them to read/write S3, write logs, etc.
resource "aws_iam_role" "emr_ec2_role" {
  name = "25nplx-emr-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Effect = "Allow"
        Principal = {Service = "ec2.amazonaws.com"}
        Action = "sts:AssumeRole"
      }]
  })

  tags = {
    Name    = "25nplx-emr-ec2-role"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Attach AWS managed policy for EMR EC2 nodes
resource "aws_iam_role_policy_attachment" "emr_ec2_policy" {
  role       = aws_iam_role.emr_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

# Instance profile wraps the role so EC2 nodes can use it
resource "aws_iam_instance_profile" "emr_ec2_profile" {
  name = "25nplx-emr-ec2-profile"
  role = aws_iam_role.emr_ec2_role.name
}

# EC2 Deployment Role 
# Allows the deployment EC2 instance to access S3
# (to download the fine-tuned GGUF model)
resource "aws_iam_role" "ec2_deployment_role" {
  name = "25nplx-ec2-deployment-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Effect = "Allow"
        Principal = {Service = "ec2.amazonaws.com"}
        Action = "sts:AssumeRole"
      }]
  })

  tags = {
    Name    = "25nplx-ec2-deployment-role"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Allow deployment EC2 to read from S3
resource "aws_iam_role_policy" "ec2_s3_policy" {
  name = "25nplx-ec2-s3-access"
  role = aws_iam_role.ec2_deployment_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
        Effect = "Allow"
        Action = ["s3:GetObject","s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::25nplx-cisc886-bucket",
          "arn:aws:s3:::25nplx-cisc886-bucket/*"
        ]
      }]
  })
}

# Instance profile for deployment EC2
resource "aws_iam_instance_profile" "ec2_deployment_profile" {
  name = "25nplx-ec2-deployment-profile"
  role = aws_iam_role.ec2_deployment_role.name
}

#  Outputs
output "emr_service_role_arn" {
  description = "EMR service role ARN — used in emr.tf"
  value       = aws_iam_role.emr_service_role.arn
}

output "emr_ec2_profile_name" {
  description = "EMR EC2 instance profile — used in emr.tf"
  value       = aws_iam_instance_profile.emr_ec2_profile.name
}

output "ec2_deployment_profile_name" {
  description = "EC2 deployment instance profile — used in deployment.tf"
  value       = aws_iam_instance_profile.ec2_deployment_profile.name
}
