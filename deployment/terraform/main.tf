# Terraform configuration for global photonic neuromorphics infrastructure
# Supports multi-cloud deployment with AWS, GCP, and Azure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket         = "photonic-neuromorphics-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "photonic-terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "photonic-neuromorphics"
}

variable "regions" {
  description = "AWS regions for multi-region deployment"
  type        = list(string)
  default     = ["us-west-2", "eu-west-1", "ap-northeast-1"]
}

variable "instance_types" {
  description = "EC2 instance types for different workloads"
  type = object({
    api     = string
    worker  = string
    gpu     = string
  })
  default = {
    api     = "c5.2xlarge"
    worker  = "c5.4xlarge"
    gpu     = "p3.2xlarge"
  }
}

variable "database_config" {
  description = "RDS database configuration"
  type = object({
    instance_class    = string
    allocated_storage = number
    engine_version   = string
  })
  default = {
    instance_class    = "db.r6g.2xlarge"
    allocated_storage = 1000
    engine_version   = "16.1"
  }
}

variable "enable_gpu_nodes" {
  description = "Enable GPU nodes in EKS cluster"
  type        = bool
  default     = true
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for HTTPS"
  type        = string
  default     = ""
}

# Local values
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Owner       = "Terragon-Labs"
  }
}

# AWS Provider Configuration
provider "aws" {
  region = "us-west-2"
  
  default_tags {
    tags = local.common_tags
  }
}

# Multi-region AWS providers
provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = local.common_tags
  }
}

provider "aws" {
  alias  = "ap_northeast_1" 
  region = "ap-northeast-1"
  
  default_tags {
    tags = local.common_tags
  }
}

# Google Cloud Provider
provider "google" {
  project = "photonic-neuromorphics-prod"
  region  = "us-west1"
}

# Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Enable VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60
  
  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
  })
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
    "kubernetes.io/role/elb"                            = 1
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-cluster" = "shared"
    "kubernetes.io/role/internal-elb"                   = 1
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name                   = "${var.project_name}-cluster"
  cluster_version               = "1.28"
  cluster_endpoint_public_access = true
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  vpc_id                   = module.vpc.vpc_id
  subnet_ids              = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.intra_subnets
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = [var.instance_types.api]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "general"
      }
      
      update_config = {
        max_unavailable_percentage = 33
      }
    }
    
    # Compute intensive nodes
    compute = {
      min_size     = 1
      max_size     = 20
      desired_size = 2
      
      instance_types = [var.instance_types.worker]
      capacity_type  = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "compute"
      }
      
      taints = {
        compute = {
          key    = "compute-intensive"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  # GPU nodes (conditional)
  dynamic "eks_managed_node_groups" {
    for_each = var.enable_gpu_nodes ? { gpu = true } : {}
    
    content {
      gpu = {
        min_size     = 0
        max_size     = 5
        desired_size = 1
        
        instance_types = [var.instance_types.gpu]
        capacity_type  = "ON_DEMAND"
        
        ami_type = "AL2_x86_64_GPU"
        
        k8s_labels = {
          Environment  = var.environment
          NodeType     = "gpu"
          "nvidia.com/gpu" = "true"
        }
        
        taints = {
          gpu = {
            key    = "nvidia.com/gpu"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        }
      }
    }
  }
  
  # aws-auth ConfigMap
  manage_aws_auth_configmap = true
  
  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "eks-admin"
      groups   = ["system:masters"]
    },
  ]
  
  tags = local.common_tags
}

# IAM Role for EKS Admin
resource "aws_iam_role" "eks_admin" {
  name = "${var.project_name}-eks-admin"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = data.aws_caller_identity.current.arn
        }
      }
    ]
  })
  
  tags = local.common_tags
}

# RDS Database
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.common_tags, {
    Name = "${var.project_name}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = local.common_tags
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_db_instance" "main" {
  identifier                = "${var.project_name}-db"
  allocated_storage         = var.database_config.allocated_storage
  max_allocated_storage     = var.database_config.allocated_storage * 2
  storage_type              = "gp3"
  storage_encrypted         = true
  
  engine         = "postgres"
  engine_version = var.database_config.engine_version
  instance_class = var.database_config.instance_class
  
  db_name  = "photonic_neuromorphics"
  username = "photonic_admin"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"
  
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.project_name}-db-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  tags = local.common_tags
}

# Redis ElastiCache
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  tags = local.common_tags
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${var.project_name}-redis"
  description                = "Redis cluster for ${var.project_name}"
  
  node_type                  = "cache.r7g.large"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = local.common_tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  enable_http2              = true
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs.id
    prefix  = "alb"
    enabled = true
  }
  
  tags = local.common_tags
}

resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = local.common_tags
}

# S3 Bucket for ALB logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${var.project_name}-alb-logs-${random_id.bucket_suffix.hex}"
  force_destroy = false
  
  tags = local.common_tags
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  rule {
    id     = "log_lifecycle"
    status = "Enabled"
    
    expiration {
      days = 90
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# CloudFront Distribution for global edge caching
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "ALB-${var.project_name}"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled             = true
  is_ipv6_enabled    = true
  comment            = "Global CDN for ${var.project_name}"
  default_root_object = "index.html"
  
  aliases = ["api.photonic-neuromorphics.ai", "www.photonic-neuromorphics.ai"]
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${var.project_name}"
    
    forwarded_values {
      query_string = true
      cookies {
        forward = "none"
      }
      headers = ["Authorization", "Content-Type"]
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl               = 0
    default_ttl           = 3600
    max_ttl               = 86400
    compress              = true
  }
  
  # API cache behavior
  ordered_cache_behavior {
    path_pattern     = "/api/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-${var.project_name}"
    
    forwarded_values {
      query_string = true
      cookies {
        forward = "all"
      }
      headers = ["*"]
    }
    
    viewer_protocol_policy = "https-only"
    min_ttl               = 0
    default_ttl           = 0
    max_ttl               = 0
  }
  
  price_class = "PriceClass_All"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn            = var.ssl_certificate_arn != "" ? var.ssl_certificate_arn : aws_acm_certificate.main[0].arn
    ssl_support_method             = "sni-only"
    minimum_protocol_version       = "TLSv1.2_2021"
  }
  
  web_acl_id = aws_wafv2_web_acl.main.arn
  
  tags = local.common_tags
}

# SSL Certificate
resource "aws_acm_certificate" "main" {
  count = var.ssl_certificate_arn == "" ? 1 : 0
  
  domain_name               = "photonic-neuromorphics.ai"
  subject_alternative_names = ["*.photonic-neuromorphics.ai"]
  validation_method         = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = local.common_tags
}

# WAF Web ACL for security
resource "aws_wafv2_web_acl" "main" {
  name  = "${var.project_name}-waf"
  scope = "CLOUDFRONT"
  
  default_action {
    allow {}
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    override_action {
      none {}
    }
    
    statement {
      rate_based_statement {
        limit              = 10000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
    
    action {
      block {}
    }
  }
  
  # Geographic restriction rule
  rule {
    name     = "GeoBlockRule"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      geo_match_statement {
        # Block traffic from specific countries if needed
        country_codes = []
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoBlockRule"
      sampled_requests_enabled   = true
    }
    
    action {
      allow {}
    }
  }
  
  tags = local.common_tags
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-waf"
    sampled_requests_enabled   = true
  }
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.configuration_endpoint_address
  sensitive   = true
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.main.domain_name
}