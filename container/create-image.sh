#!/bin/bash
set -euo pipefail

is_sagemaker_studio() {
    # Defaults keep detection safe when set -u is enabled.
    if [[ -n "${SM_CURRENT_HOST:-}" ]] ||
       [[ -n "${SAGEMAKER_INTERNAL_IMAGE_URI:-}" ]] ||
       [[ -n "${SM_USER_ID:-}" ]]; then
        return 0
    fi
    if [[ -d "/opt/ml" ]] && [[ -f "/opt/ml/metadata/resource-metadata.json" ]]; then
        return 0
    fi
    if [[ -f "/.dockerenv" ]] && [[ $(hostname) =~ ^sagemaker-* ]]; then
        return 0
    fi
    if [[ $(whoami) == "sagemaker-user" ]]; then
        return 0
    fi
    return 1
}

if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <REPO_NAME> [TAG] [DOCKERFILE] [CONTEXT] [docker build options...]"
    echo "Defaults: TAG=latest DOCKERFILE=Dockerfile CONTEXT=."
    echo "Paths are relative to the current working directory."
    echo "Example: $0 my-image latest Dockerfile . --build-arg VERSION=1.0"
    echo "PLATFORM defaults to linux/amd64; set it to select another target architecture."
    echo "Authenticate to private base-image registries before running this script."
    echo "This script builds the image, creates the destination ECR repository if missing, and pushes."
    if [ "$#" -eq 0 ]; then exit 1; fi
    exit 0
fi

REPO_NAME=$1
shift
TAG=${1:-latest}
if [ "$#" -gt 0 ]; then shift; fi
DOCKERFILE=${1:-Dockerfile}
if [ "$#" -gt 0 ]; then shift; fi
BUILD_CONTEXT=${1:-.}
if [ "$#" -gt 0 ]; then shift; fi

if [ ! -f "$DOCKERFILE" ]; then
    echo "Dockerfile not found: $DOCKERFILE" >&2
    exit 1
fi

AWS_REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-}}
if [ -z "$AWS_REGION" ]; then
    AWS_REGION=$(aws configure get region) || {
        echo "Set AWS_REGION or AWS_DEFAULT_REGION, or configure an AWS CLI region." >&2
        exit 1
    }
fi
: "${AWS_REGION:?Could not determine AWS region}"
export AWS_REGION

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if ! [[ "$ACCOUNT" =~ ^[0-9]{12}$ ]]; then
    echo "Invalid AWS account returned by STS: $ACCOUNT" >&2
    exit 1
fi
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"

# Authenticate before building so private base images in this registry also work.
aws ecr get-login-password --region "$AWS_REGION" |
    docker login --username AWS --password-stdin "$REGISTRY"

echo "Building $IMAGE_URI using $DOCKERFILE (context: $BUILD_CONTEXT)"
if is_sagemaker_studio; then
    echo "Detected SageMaker Studio environment - using --network sagemaker"
    docker build --network sagemaker --platform "${PLATFORM:-linux/amd64}" \
        "$@" -f "$DOCKERFILE" -t "$IMAGE_URI" "$BUILD_CONTEXT"
else
    echo "Detected local/standard environment - using default network"
    docker build --platform "${PLATFORM:-linux/amd64}" \
        "$@" -f "$DOCKERFILE" -t "$IMAGE_URI" "$BUILD_CONTEXT"
fi

# Only a missing repository warrants creation; permission errors do not.
if REPO_RESULT=$(aws ecr describe-repositories \
    --repository-names "$REPO_NAME" --region "$AWS_REGION" 2>&1); then
    echo "Repository $REPO_NAME already exists"
else
    case "$REPO_RESULT" in
        *RepositoryNotFoundException*)
            aws ecr create-repository --repository-name "$REPO_NAME" \
                --region "$AWS_REGION"
            ;;
        *)
            printf '%s\n' "$REPO_RESULT" >&2
            exit 1
            ;;
    esac
fi

docker image push "$IMAGE_URI"
printf 'Image pushed successfully: %s\n' "$IMAGE_URI"
