# Workshop AWS Infrastructure

`workshop.yaml` creates the shared AWS prerequisites used by the workshop. It
does not create GPU instances, SageMaker Training Jobs, EKS clusters, HyperPod
clusters, or endpoints, so deploying the stack does not start compute billing.

## Resources

- A private, encrypted, versioned S3 bucket for datasets, configuration,
  checkpoints, and exported models.
- A shared `WorkshopExecutionRole` with:
  - `sagemaker.amazonaws.com` trust for SageMaker Training Jobs;
  - `pods.eks.amazonaws.com` trust for EKS Pod Identity;
  - `ec2.amazonaws.com` trust plus an instance profile for EC2-backed workers
    and HyperPod Slurm nodes;
  - scoped access to the workshop bucket;
  - optional ECR pull/push access;
  - CloudWatch logging/metrics and SageMaker training ENI permissions;
  - KMS and Lambda runtime integration required by `ModelTrainer`;
  - SageMaker Hub, Model Registry, MLflow, lineage, and evaluation permissions;
  - Amazon Bedrock discovery and model-invocation permissions.
- An optional encrypted ECR repository for the custom training image.
- A Lambda-backed custom resource that empties the versioned S3 bucket before
  CloudFormation deletes it.

The bucket cleanup Lambda has a separate least-privilege role. The workload
role is intentionally not allowed to perform stack cleanup.

## Deploy

Verify the caller and region first:

```bash
aws sts get-caller-identity
aws configure get region
```

Deploy from the repository root:

```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/workshop.yaml \
  --stack-name physical-ai-workshop \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ProjectName=physical-ai-workshop \
    CreateEcrRepository=true \
    EcrRepositoryName=smolvla-training
```

Inspect the values consumed by the notebooks and launchers:

```bash
aws cloudformation describe-stacks \
  --stack-name physical-ai-workshop \
  --query "Stacks[0].Outputs"
```

The identity launching a SageMaker Training Job still needs
`cloudformation:DescribeStacks`, `sagemaker:CreateTrainingJob`, and
`iam:PassRole` for the emitted `WorkshopExecutionRoleArn`. Those are caller
permissions, not permissions the training container needs after SageMaker
assumes the role.

## Use with EKS and HyperPod

The output role is a workload data/model role:

- For EKS or HyperPod EKS, associate it with a Kubernetes service account using
  EKS Pod Identity.
- For EC2-backed workers or HyperPod Slurm, use the emitted instance profile.
- For SageMaker managed jobs, pass `WorkshopExecutionRoleArn` as the execution
  role.

The base stack deliberately does not create an EKS cluster, VPC, FSx file
system, or HyperPod cluster. HyperPod cluster provisioning normally requires a
separate control-plane execution role and additional networking permissions.
In production, split service roles further; the shared role here keeps the
workshop portable without claiming that one role is an entire cluster setup.

Example EKS Pod Identity association:

```bash
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name physical-ai-workshop \
  --query "Stacks[0].Outputs[?OutputKey=='WorkshopExecutionRoleArn'].OutputValue" \
  --output text)

aws eks create-pod-identity-association \
  --cluster-name <cluster-name> \
  --namespace <namespace> \
  --service-account <service-account> \
  --role-arn "$ROLE_ARN"
```

## Delete

```bash
aws cloudformation delete-stack --stack-name physical-ai-workshop
aws cloudformation wait stack-delete-complete --stack-name physical-ai-workshop
```

Deletion is destructive:

- the custom resource permanently removes every current object, noncurrent
  version, delete marker, and multipart upload from the S3 bucket;
- `EmptyOnDelete` permanently removes images from the optional ECR repository;
- CloudFormation then deletes the bucket, repository, roles, and Lambda.

The cleanup Lambda has a 15-minute timeout and is intended for workshop-sized
buckets. Large production datasets should use a retention policy or a separate
S3 Batch Operations cleanup workflow instead.

GPU quotas are not created by CloudFormation. Before Notebook 3, verify the
SageMaker training quota for the selected instance type. HyperPod uses separate
cluster quotas and continues billing until its cluster is stopped or deleted.
