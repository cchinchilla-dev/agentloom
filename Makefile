.PHONY: install install-all test test-cov lint format typecheck check run validate info build clean \
       docker-build docker-build-obs docker-run docker-stack docker-stack-down \
       k8s-validate helm-lint helm-template

# Development
install:
	uv sync --group dev

install-all:
	uv sync --group dev --all-extras

# Quality
test:
	uv run pytest

test-cov:
	uv run pytest --cov=agentloom --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint typecheck test

# CLI shortcuts
run:
	@test -n "$(WORKFLOW)" || (echo "Usage: make run WORKFLOW=examples/01_simple_qa.yaml" && exit 1)
	uv run agentloom run $(WORKFLOW)

validate:
	@test -n "$(WORKFLOW)" || (echo "Usage: make validate WORKFLOW=examples/01_simple_qa.yaml" && exit 1)
	uv run agentloom validate $(WORKFLOW)

info:
	uv run agentloom info

# Build
build:
	uv build --wheel

# Docker
docker-build:
	docker build -t agentloom:local .

docker-build-obs:
	docker build --build-arg BUILD_OBSERVABILITY=true -t agentloom:local-obs .

docker-run:
	@test -n "$(WORKFLOW)" || (echo "Usage: make docker-run WORKFLOW=01_simple_qa.yaml" && exit 1)
	docker run --rm -v $(CURDIR)/examples:/workflows:ro agentloom:local run /workflows/$(WORKFLOW)

docker-stack:
	cd deploy && docker compose up -d

docker-stack-down:
	cd deploy && docker compose down

# Kubernetes
k8s-validate:
	kustomize build deploy/k8s/overlays/dev | kubeconform -strict -summary
	kustomize build deploy/k8s/overlays/staging | kubeconform -strict -summary
	kustomize build deploy/k8s/overlays/production | kubeconform -strict -summary

helm-lint:
	helm lint deploy/helm/agentloom -f deploy/helm/agentloom/ci/test-values.yaml

helm-template:
	helm template test deploy/helm/agentloom -f deploy/helm/agentloom/ci/test-values.yaml -n agentloom

# Cleanup
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
