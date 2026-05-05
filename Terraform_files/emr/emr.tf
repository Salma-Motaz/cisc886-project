# ============================================================
# emr.tf - EMR Cluster (Automated spark-submit)
# NetID: 25nplx
#
# LIFECYCLE:
#   1. cd emr && terraform init && terraform apply
#   2. Monitor in AWS Console → EMR → Clusters
#   3. Wait for Spark job to finish
#   4. terraform destroy  → terminate cluster immediately
# ============================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}


# Data Sources

data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["25nplx-vpc"]
  }
  filter {
    name   = "cidr"
    values = ["10.0.0.0/16"]
  }
}

data "aws_subnet" "private" {
  filter {
    name   = "tag:Name"
    values = ["25nplx-subnet-private"]
  }
}

data "aws_iam_role" "emr_service_role" {
  name = "25nplx-emr-service-role"
}

data "aws_iam_instance_profile" "emr_ec2_profile" {
  name = "25nplx-emr-ec2-profile"
}


# EMR Master Security Group
# SSH open for debugging if Spark job fails
# Restricted to VPC CIDR — not public internet

resource "aws_security_group" "emr_master" {
  name        = "25nplx-sg-emr-master"
  description = "EMR master node security group"
  vpc_id = data.aws_vpc.main.id
  revoke_rules_on_delete = true

  # SSH kept open for debugging if job fails
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "SSH from within VPC only"
  }

  # All outbound - needed to access S3 and download packages
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "25nplx-sg-emr-master", Project = "cisc886", NetID = "25nplx" }
}

# EMR Core Security Group
# Workers only accept traffic from master node
# No public access needed
resource "aws_security_group" "emr_core" {
  name        = "25nplx-sg-emr-core"
  description = "EMR core/worker nodes - only talk to master"
  vpc_id = data.aws_vpc.main.id
  revoke_rules_on_delete = true

  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.emr_master.id]
    description     = "Traffic from master node only"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "25nplx-sg-emr-core", Project = "cisc886", NetID = "25nplx" }
}

# EMR Service Security Group 
# Required for EMR in private subnet
# Port 9443 allows master to communicate with AWS EMR service

resource "aws_security_group" "emr_service" {
  name   = "25nplx-sg-emr-service"
  vpc_id = data.aws_vpc.main.id
  revoke_rules_on_delete = true

  # Port 9443 required for EMR master to communicate with AWS service
  ingress {
    from_port       = 9443
    to_port         = 9443
    protocol        = "tcp"
    security_groups = [aws_security_group.emr_master.id]
    description     = "EMR master to service communication"
  }

  ingress {
  from_port   = 0
  to_port     = 0
  protocol    = "-1"
  cidr_blocks = ["10.0.0.0/16"]
  description = "VPC internal traffic"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "25nplx-sg-emr-service"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# EMR Cluster
resource "aws_emr_cluster" "spark_cluster" {
  name          = "25nplx-emr"
  release_label = "emr-6.15.0"
  applications  = ["Spark", "Hadoop"]

  ec2_attributes {
    subnet_id                         = data.aws_subnet.private.id
    emr_managed_master_security_group = aws_security_group.emr_master.id
    emr_managed_slave_security_group  = aws_security_group.emr_core.id
    service_access_security_group     = aws_security_group.emr_service.id
    instance_profile                  =  data.aws_iam_instance_profile.emr_ec2_profile.name
    key_name                          = "25nplx-keypair"
    
  }
  
  ebs_root_volume_size = 50

  # Master node - coordinates the Spark job across workers
  master_instance_group {
    instance_type = "m5.xlarge"   # 4 vCPU, 16 GB RAM
  }

  # 2 worker nodes - process Ubuntu corpus in parallel
  # splitting data across 2 machines = 2x faster processing
  core_instance_group {
    instance_type  = "m5.xlarge"  # 4 vCPU, 16 GB RAM each
    instance_count = 2
  }
  
  service_role = data.aws_iam_role.emr_service_role.arn


  # Save EMR logs to S3 - useful for debugging if job fails
  log_uri = "s3://25nplx-cisc886-bucket/emr-logs/"

  bootstrap_action {
    name = "Install Python dependencies"
    path = "s3://25nplx-cisc886-bucket/scripts/bootstrap.sh"
  }

  # Automated Spark Step 
  step {
    name              = "25nplx-ubuntu-preprocessing"
    action_on_failure = "CONTINUE"  # keep cluster alive if step fails for debug logs

    hadoop_jar_step {
      jar = "command-runner.jar"  # built-in EMR tool for running spark-submit
      args = [
        "spark-submit",
        "--deploy-mode", "cluster",           # run job ON the cluster not locally
        "--conf", "spark.executor.memory=8g", # each worker gets 6GB RAM
        "--conf", "spark.driver.memory=4g",   # coordinator gets 2GB RAM
        "--conf", "spark.executor.cores=4",   # use all 4 cores per worker
        "--conf", "spark.yarn.appMasterEnv.PYSPARK_PYTHON=python3",
        "--conf", "spark.executorEnv.PYSPARK_PYTHON=python3",
        "s3://25nplx-cisc886-bucket/scripts/ubuntu_preprocessing.py"
      ]
    }
  }

  # terminate the cluster immediately when step finishes
  keep_job_flow_alive_when_no_steps = false

  # Backup safety net - terminate after 30 min if step gets stuck
  #auto_termination_policy {
  #  idle_timeout = 1800  # 30 minutes in seconds
  #}

  tags = { Name = "25nplx-emr", Project = "cisc886", NetID = "25nplx" }
}

# Outputs
output "emr_cluster_id" {
  description = "Cluster ID - monitor progress in AWS console"
  value       = aws_emr_cluster.spark_cluster.id
}

output "s3_output_path" {
  description = "Check here for preprocessed output files after job"
  value       = "s3://25nplx-cisc886-bucket/processed/"
}

output "s3_logs_path" {
  description = "Check here if job fails"
  value       = "s3://25nplx-cisc886-bucket/emr-logs/"
}
