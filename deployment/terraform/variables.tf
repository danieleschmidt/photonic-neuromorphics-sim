# Variables for Photonic Neuromorphics Infrastructure
variable "aws_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be production, staging, or development."
  }
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "photonic-neuromorphics"
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.28"
}

variable "node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    min_size       = number
    max_size       = number
    desired_size   = number
    capacity_type  = string
  }))
  default = {
    general = {
      instance_types = ["c5.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      capacity_type  = "ON_DEMAND"
    }
    compute = {
      instance_types = ["c5.4xlarge"]
      min_size       = 1
      max_size       = 20
      desired_size   = 2
      capacity_type  = "SPOT"
    }
    gpu = {
      instance_types = ["p3.2xlarge"]
      min_size       = 0
      max_size       = 5
      desired_size   = 1
      capacity_type  = "ON_DEMAND"
    }
  }
}

variable "database_config" {
  description = "RDS PostgreSQL configuration"
  type = object({
    instance_class    = string
    allocated_storage = number
    engine_version   = string
    backup_retention = number
  })
  default = {
    instance_class    = "db.r6g.2xlarge"
    allocated_storage = 1000
    engine_version   = "16.1"
    backup_retention  = 30
  }
}

variable "redis_config" {
  description = "ElastiCache Redis configuration"
  type = object({
    node_type         = string
    num_cache_clusters = number
    parameter_group   = string
  })
  default = {
    node_type          = "cache.r7g.large"
    num_cache_clusters = 2
    parameter_group    = "default.redis7"
  }
}

variable "enable_gpu_nodes" {
  description = "Enable GPU-enabled node groups"
  type        = bool
  default     = true
}

variable "enable_multi_az" {
  description = "Enable multi-AZ deployment for RDS and Redis"
  type        = bool
  default     = true
}

variable "ssl_certificate_arn" {
  description = "ARN of existing SSL certificate (optional)"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
  default     = "photonic-neuromorphics.ai"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
  
  validation {
    condition     = var.availability_zones >= 2 && var.availability_zones <= 4
    error_message = "Number of availability zones must be between 2 and 4."
  }
}

variable "log_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 30
}

variable "backup_retention_days" {
  description = "Database backup retention period in days"
  type        = number
  default     = 30
}

variable "monitoring_config" {
  description = "Monitoring and alerting configuration"
  type = object({
    enable_detailed_monitoring = bool
    enable_performance_insights = bool
    enable_enhanced_monitoring = bool
  })
  default = {
    enable_detailed_monitoring  = true
    enable_performance_insights = true
    enable_enhanced_monitoring  = true
  }
}

variable "security_config" {
  description = "Security configuration settings"
  type = object({
    enable_waf                = bool
    enable_vpc_flow_logs     = bool
    enable_cloudtrail        = bool
    enable_config            = bool
    enable_guardduty         = bool
  })
  default = {
    enable_waf            = true
    enable_vpc_flow_logs  = true
    enable_cloudtrail     = true
    enable_config         = true
    enable_guardduty      = true
  }
}

variable "cost_allocation_tags" {
  description = "Additional tags for cost allocation"
  type        = map(string)
  default = {
    CostCenter = "research-development"
    Department = "ai-ml"
    Team       = "terragon-labs"
  }
}

variable "auto_scaling_config" {
  description = "Auto scaling configuration for node groups"
  type = object({
    scale_up_cooldown   = number
    scale_down_cooldown = number
    target_cpu_utilization = number
  })
  default = {
    scale_up_cooldown      = 300
    scale_down_cooldown    = 300
    target_cpu_utilization = 70
  }
}