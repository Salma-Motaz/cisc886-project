# ============================================================
# main.tf — Network Foundation
# Project: CISC886 Cloud Computing
# NetID: 25nplx
# Purpose: VPC, subnet, internet gateway, route table
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

# VPC 
# 10.0.0.0/16 gives 65,536 IPs — plenty of room for EMR
# nodes and EC2 instances without reconfiguration later
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true   # required for EMR to resolve hostnames
  enable_dns_support   = true

  tags = {
    Name    = "25nplx-vpc"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Public Subnet
# Single public subnet in us-east-1a
# Hosts: bastion host, NAT gateway
# map_public_ip_on_launch = true so bastion gets public IP
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true  # auto-assign public IP to all instances

  tags = {
    Name    = "25nplx-subnet-public"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}


# Private Subnet
# Single private subnet in us-east-1a
# Hosts: EMR cluster, LLM EC2 instance
# No public IPs — only reachable via bastion or NAT
resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-east-1a"
  map_public_ip_on_launch = false
  
  tags = {
    Name    = "25nplx-subnet-private"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Internet Gateway 
# Allows public subnet to reach internet
# Required for bastion SSH and NAT gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name    = "25nplx-igw"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}


# Elastic IP for NAT
# Fixed public IP address for the NAT gateway
resource "aws_eip" "nat" {
  domain = "vpc"
}

# NAT Gateway (must be in PUBLIC subnet)
# Allows private subnet instances to reach internet
# outbound only — EMR downloads packages, EC2 pulls model
# Must live in PUBLIC subnet
resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id

  depends_on = [aws_internet_gateway.igw]

  tags = {
    Name = "25nplx-nat"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Route Table 
# All traffic from public subnet goes to internet via IGW
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name    = "25nplx-rt-public"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Private Route Table
# All traffic from private subnet goes to internet via NAT
# NAT allows outbound only — no inbound from internet
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }

  tags = {
    Name = "25nplx-rt-private"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Route Table Association 
# Links the route table to the public, and private subnet so the
# routing rules above apply to all instances in that subnet
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private.id
}

# S3 VPC Endpoint
# Allows EMR to read/write S3 directly without going
# through NAT Gateway — faster and free (no NAT charges)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.us-east-1.s3"

  route_table_ids = [
  aws_route_table.private.id,
  aws_route_table.public.id
  ]

  depends_on = [
  aws_route_table.private,
  aws_route_table.public
  ]

  tags = {
    Name = "25nplx-s3-endpoint"
  }
}


# Outputs 
# Used by emr/ and deployment/ folders via data sources
output "vpc_id" {
  description = "VPC ID — used by EMR and EC2"
  value       = aws_vpc.main.id
}

output "public_subnet_id" {
  description = "Public subnet ID — used for internet-facing resources"
  value = aws_subnet.public.id
}

output "private_subnet_id" {
  description = "Private subnet ID — used by EMR and EC2"
  value = aws_subnet.private.id
}

output "availability_zone" {
  description = "AZ of the public subnet"
  value       = aws_subnet.public.availability_zone
}

