# AWS Deployment Plan

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Users                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CloudFront (CDN + HTTPS)                       │
│         ┌─────────────┐         ┌─────────────────┐        │
│         │  S3 Bucket  │         │  ALB (HTTPS)    │        │
│         │  (frontend) │         │                 │        │
│         └─────────────┘         └────────┬────────┘        │
└──────────────────────────────────────────┼──────────────────┘
                                           │
                                           ▼
                           ┌─────────────────────────────┐
                           │   ECS Fargate Cluster       │
                           │   ┌─────────────────────┐   │
                           │   │  FastAPI Container  │   │
                           │   │  (2-4 tasks, auto)  │   │
                           │   └─────────────────────┘   │
                           └─────────────┬───────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │ RDS Postgres │    │ ElastiCache  │    │  GLM-4 API   │
            │ + pgvector   │    │ Redis        │    │  (external)  │
            └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Pre-Requisites

1. AWS CLI installed and configured (`aws configure`)
2. Docker installed
3. Domain name (optional — can use ALB DNS initially)

---

## Step 1: Environment Setup

Copy `.env.example` to `.env` and fill in your secrets:

```bash
cd apps/backend
cp .env.example .env
# Edit .env with your actual API keys and DB credentials
```

---

## Step 2: Database (RDS PostgreSQL)

1. **Create RDS instance**:
   - Engine: PostgreSQL 16
   - Instance class: `db.t3.micro` (dev) or `db.t3.small` (prod)
   - Enable public accessibility (for initial setup, then restrict)
   - Set master username/password
   - Note the endpoint: `your-db.xxx.us-east-1.rds.amazonaws.com`

2. **Enable pgvector extension**:
   ```bash
   psql -h your-db.xxx.us-east-1.rds.amazonaws.com -U postgres -d ecommerce
   CREATE EXTENSION vector;
   \q
   ```

3. **Update `.env`**:
   ```bash
   DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@your-db.xxx.us-east-1.rds.amazonaws.com:5432/ecommerce
   ```

4. **Run migrations locally** (with AWS DB accessible):
   ```bash
   source venv/bin/activate
   python -m backend.db.migrate_pgvector
   ```

---

## Step 3: Redis (ElastiCache)

1. **Create ElastiCache Redis** cluster:
   - Node type: `cache.t3.micro`
   - Same VPC as ECS tasks
   - No cluster mode (single node)

2. **Update `.env`**:
   ```bash
   REDIS_URL=redis://your-redis.xxx.cache.amazonaws.com:6379/0
   ```

---

## Step 4: Containerize & Push to ECR

1. **Create ECR repository**:
   ```bash
   aws ecr create-repository --repository-name ecommerce-agent-backend --region us-east-1
   ```

2. **Login to ECR**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   ```

3. **Build and push**:
   ```bash
   cd apps/backend
   docker build -t ecommerce-agent-backend .
   docker tag ecommerce-agent-backend:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ecommerce-agent-backend:latest
   docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ecommerce-agent-backend:latest
   ```

---

## Step 5: ECS Fargate

### 5.1 Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name ecommerce-agent-cluster --region us-east-1
```

### 5.2 Create Task Definition

Create `task-definition.json`:

```json
{
  "family": "ecommerce-agent-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ecommerce-agent-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DATABASE_URL", "value": "postgresql+psycopg2://..."},
        {"name": "OPENAI_API_KEY", "value": "your-key"},
        {"name": "OPENAI_API_BASE", "value": "https://open.bigmodel.cn/api/paas/v4/"},
        {"name": "REDIS_URL", "value": "redis://..."},
        {"name": "CORS_ORIGINS", "value": "https://yourdomain.com"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ecommerce-agent",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register it:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### 5.3 Create Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer --name ecommerce-agent-alb --subnets subnet-xxx subnet-yyy --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group --name ecommerce-agent-tg --protocol HTTP --port 8000 --vpc-id vpc-xxx --target-type ip

# Create listener
aws elbv2 create-listener --load-balancer-arn arn:aws:elasticloadbalancing:... --protocol HTTPS --port 443 --certificates CertificateArn=arn:aws:acm:... --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### 5.4 Create ECS Service

```bash
aws ecs create-service \
  --cluster ecommerce-agent-cluster \
  --service-name ecommerce-agent-service \
  --task-definition ecommerce-agent-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=backend,containerPort=8000"
```

### 5.5 Enable Auto-Scaling

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/ecommerce-agent-cluster/ecommerce-agent-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --policy-name cpu-autoscaling \
  --service-namespace ecs \
  --resource-id service/ecommerce-agent-cluster/ecommerce-agent-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration "targetValue=70.0,predefinedMetricSpecification={predefinedMetricType=ECSServiceAverageCPUUtilization}"
```

---

## Step 6: Frontend (S3 + CloudFront)

1. **Build the frontend**:
   ```bash
   cd apps/frontend
   npm run build
   ```

2. **Create S3 bucket**:
   ```bash
   aws s3 mb s3://your-ecommerce-agent-frontend
   aws s3 website s3://your-ecommerce-agent-frontend --index-document index.html --error-document index.html
   ```

3. **Upload build**:
   ```bash
   aws s3 sync dist/ s3://your-ecommerce-agent-frontend/ --delete
   ```

4. **Create CloudFront distribution**:
   - Origin: S3 bucket
   - Default root object: `index.html`
   - Enable HTTPS with ACM certificate
   - Cache behaviors: default → S3, `/api/*` → ALB origin

5. **Update frontend API base URL** to point to your ALB/CloudFront domain.

---

## Step 7: Security

### Security Groups

| Service | Ingress | Source |
|---------|---------|--------|
| ALB | 80, 443 | 0.0.0.0/0 |
| ECS tasks | 8000 | ALB security group |
| RDS | 5432 | ECS tasks security group |
| ElastiCache | 6379 | ECS tasks security group |

### Secrets Manager (Recommended)

Instead of putting secrets in task definition environment variables, use AWS Secrets Manager:

```bash
aws secretsmanager create-secret --name ecommerce-agent/prod --secret-string file://.env
```

Then reference in task definition via `secrets` field.

---

## Step 8: Monitoring

1. **CloudWatch Logs**: ECS task logs automatically go to `/ecs/ecommerce-agent`
2. **CloudWatch Metrics**: Set alarms for:
   - CPU utilization > 80%
   - Memory utilization > 80%
   - 5xx error rate > 1%
   - ALB target response time > 5s

3. **Structured Logging** (optional upgrade):
   Add `python-json-logger` and output JSON logs for CloudWatch Insights queries.

---

## Cost Estimate (Monthly)

| Service | Instance | Cost |
|---------|----------|------|
| ECS Fargate | 2 tasks × 1 vCPU / 2GB | ~$35 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ElastiCache Redis | cache.t3.micro | ~$15 |
| ALB | | ~$20 |
| S3 + CloudFront | | ~$5 |
| GLM-4 API | ~1k requests/day | ~$10-30 |
| **Total** | | **~$100-120/month** |

---

## Local Testing with Docker

Before deploying to AWS, test locally:

```bash
cd apps/backend

# Build
docker build -t ecommerce-agent-backend .

# Run with .env
docker run -p 8000:8000 --env-file .env ecommerce-agent-backend

# Test
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"message":"hello"}'
```

---

## CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push image
        run: |
          docker build -t ecommerce-agent-backend ./apps/backend
          docker tag ecommerce-agent-backend:latest ${{ steps.login-ecr.outputs.registry }}/ecommerce-agent-backend:latest
          docker push ${{ steps.login-ecr.outputs.registry }}/ecommerce-agent-backend:latest

      - name: Update ECS service
        run: |
          aws ecs update-service --cluster ecommerce-agent-cluster --service ecommerce-agent-service --force-new-deployment
```

---

## Rollback Plan

If a deployment breaks:

```bash
# Rollback to previous task definition
aws ecs update-service \
  --cluster ecommerce-agent-cluster \
  --service ecommerce-agent-service \
  --task-definition ecommerce-agent-task:PREVIOUS_REVISION \
  --force-new-deployment
```

---

## Next Steps / Future Improvements

1. **Redis session memory**: Replace JSON file memory store with Redis for concurrent safety
2. **Rate limiting**: Add `slowapi` with Redis backend
3. **True LLM streaming**: Wire `agent_graph.astream()` into `/chat/stream`
4. **Database migrations**: Add Alembic for schema versioning
5. **Multi-region**: Deploy ECS + RDS read replicas in multiple regions
6. **Load testing**: Use `locust` or `k6` to find real concurrency limits
