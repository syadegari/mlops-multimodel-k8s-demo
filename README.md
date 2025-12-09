- [MLOps Demo – Multi-Model Inference on Kubernetes](#mlops-demo--multi-model-inference-on-kubernetes)
  - [1. Objectives](#1-objectives)
  - [2. Scenario Overview](#2-scenario-overview)
    - [Models](#models)
  - [3. High-Level Overview](#3-high-level-overview)
  - [4. Containers](#4-containers)
    - [4.1 Shared parts of the design](#41-shared-parts-of-the-design)
    - [4.2 Causal LM Service Image (Heavy)](#42-causal-lm-service-image-heavy)
    - [4.3 sklearn Classifier Service Image (Light)](#43-sklearn-classifier-service-image-light)
  - [5. Kubernetes Deployment Model](#5-kubernetes-deployment-model)
    - [5.1 Basic Resources](#51-basic-resources)
    - [5.2 Horizontal Pod Autoscaler (HPA)](#52-horizontal-pod-autoscaler-hpa)
  - [6. Artificial Load \& Observability](#6-artificial-load--observability)
    - [6.1 Artificial Load](#61-artificial-load)
    - [6.2 Metrics \& Logging](#62-metrics--logging)
  - [7. API Design \& Documentation](#7-api-design--documentation)
    - [Example Endpoints](#example-endpoints)
  - [8. Demo Runbook (Reproducible Steps)](#8-demo-runbook-reproducible-steps)
    - [8.1 Prerequisites](#81-prerequisites)
    - [8.2 Start minikube (once per session)](#82-start-minikube-once-per-session)
    - [8.3 Train and register classifier models](#83-train-and-register-classifier-models)
    - [8.4 Build Docker images for both services](#84-build-docker-images-for-both-services)
    - [8.5 Load images into minikube](#85-load-images-into-minikube)
    - [8.6 Mount the models directory into minikube](#86-mount-the-models-directory-into-minikube)
    - [8.7 Deploy classifier and LM services to Kubernetes](#87-deploy-classifier-and-lm-services-to-kubernetes)
    - [8.8 Test both services](#88-test-both-services)
      - [8.8.1 Classifier service](#881-classifier-service)
      - [8.8.2 LM service](#882-lm-service)
    - [8.9 Autoscaling demos](#89-autoscaling-demos)
      - [8.9.1 Classifier autoscaling demo](#891-classifier-autoscaling-demo)
      - [8.9.2 LM autoscaling demo](#892-lm-autoscaling-demo)
      - [8.9.3 Sample output](#893-sample-output)
    - [8.10 Cleanup](#810-cleanup)

# MLOps Demo – Multi-Model Inference on Kubernetes 

## 1. Objectives

The following goals are considered for this demo:
  - Design, deploy and operate scalable inference services for ML models.
  - Package each model model as a separate, reproducible container.
  - Run them on a Kubernetes cluster with autoscaling and monitoring.
  - Create and expose clean APIs that can be used by non-ML users.
  - Versioning of trained models and ability to roll back as well as providing metrics and logging.
This demo is a smale-scale, single node minikube, but the overall architecture is flexible enough that can be scaled to production node/hardware.

---

## 2. Scenario Overview

### Models

State of art models and/or heavy models are deliberately avoided to focus on MLOps patterns and constrast between a heavy and lightweight model.

- **Model A – Causal LM service (simulating a heavy model)**\
  A small **autoregressive language model** (e.g. `gpt2` or another small `AutoModelForCausalLM` from Hugging Face) served via FastAPI.

  - Task: answer short text questions or generate short text replies given an input prompt (capped to \~256 tokens total).

- **Model B – sklearn classifier on MNIST (Simulating a light model)**\
  A lightweight **scikit-learn classifier** (e.g. RandomForestClassifier) trained on the MNIST digits dataset.

  - Task: classify an input image (or pre-extracted features) as one of the 10 MNIST digits.

---

## 3. High-Level Overview

- A single **Kubernetes cluster** (minikube) running on a local machine.
- Two independent microservices each in its own pod:
  - `lm-service` (causal LM, heavy image, CPU-only for the demo).
  - `classifier-service` (sklearn model, very small image).

- For each service there is a dedicated set of Kubernetes resources:
  - one Docker image (lm-service:v1, classifier-service:v1).
  - one Deployment (with resource requests/limits).
  - one Service of type ClusterIP.
  - one Horizontal Pod Autoscaler (HPA) driven by CPU utilisation.

- The classifier service mounts the host models/ directory into the pod as /models, so the trained `.joblib` artifacts act as a simple local model registry; the `MODEL_PATH` environment variable selects which artifact to load at startup.

- Access from the host machine is via `kubectl port-forward` to each Service.

This design aschieves isolation, independent scaling, and clear separation of responsibilities.

---

## 4. Containers

### 4.1 Shared parts of the design

- Each model lives in its own Docker image and is inferrence only.
- Build arguments or environment variables carry the model version.
- Images are tagged: `lm-service:v1`, `lm-service:v2`, `classifier-service:v1`, etc.

### 4.2 Causal LM Service Image (Heavy)

- Base image: a Transformers-capable (CPU-only) Python environment:

  ```Dockerfile
  FROM huggingface/transformers-pytorch-cpu
  RUN pip install fastapi uvicorn[standard]
  ```

- The image:
  - Loads a small `AutoModelForCausalLM` + tokenizer at startup.
  - Exposes FastAPI endpoints:
    - `GET /healthz` – health check.
    - `POST /generate` – accepts JSON with `prompt`, `max_new_tokens`, etc., and returns generated text.
  - Enforces a token budget, such that prompt and response together are less than 256 tokens.

### 4.3 sklearn Classifier Service Image (Light)

- Model training (baked into the project):

  - A separate training script (in `./services/classifier-service/train_mnist_rf.py`) lives in the repository.
  - It loads the MNIST dataset, splits into train/test, trains a RandomForest classifier with a few configurable hyperparameters (number of trees, max depth, train/test ratio), and evaluates it.
  - It saves the model as a versioned artifact, e.g. `./models/mnist_rf_v1.joblib`, `./models/mnist_rf_v2.joblib`, etc.

- Runtime model selection via mounted volume:

  - At runtime, Kubernetes mounts a host directory (e.g. `./models/` directory) into the pod at `/models` (as a simple model registry).
  - An environment variable (e.g. `MODEL_PATH`) tells the app which file inside the mounted directory to load.
  - To switch models, we need to change `MODEL_PATH` in the Deployment (e.g. to `/models/mnist_rf_v2.joblib`)

- The service exposes endpoints:

  - `GET /healthz` – health check.
  - `POST /predict` – accepts simple feature vectors and returns a class + probability.

---

## 5. Kubernetes Deployment Model

### 5.1 Basic Resources

For each service (`lm-service`, `classifier-service`):

- `Deployment` and `service` specifying:
  - Container image + tag.
  - Resource requests and limits (more CPU/memory for `lm-service`).
  - Environment variables, including `MODEL_VERSION`.

### 5.2 Horizontal Pod Autoscaler (HPA)

- Configure an HPA for each `Deployment` based on CPU utilization:
  - Example for `classifier-service`: target 70% CPU, min 1 pod, max 5 pods.

---

## 6. Artificial Load & Observability

### 6.1 Artificial Load

In order to observe Horizontal Pod Autoscaling (HPA), we need to inject artificial load and add basic observability. This is managed by two scripts `load_test_classifier.py` and `load_test_lm.py` for each deployed pod. Each script adds a time sleep in the request handler and performs a small dummy CPU loop that can be controlled with `ARTIFICIAL_CPU_LOAD` that is located in `deployment.yaml` of each pod, respectively.

### 6.2 Metrics & Logging

- Each load test script measures the following (application level) metrics:
  - Request latency (start/end timestamps).
  - Request count, error count.
- Kubernetes-level metrics (CPU and memory usage) can be measured with `kubectl top pods`
- Structured JSON logs from FastAPI (request id, path, latency, status).

---

## 7. API Design & Documentation

### Example Endpoints

- `lm-service`:

  - `POST /generate` –
    - Input: `{ "prompt": "Describe language models in one sentence", "max_new_tokens": 64 }`
    - Output: `{ "generated_text": "..." }`

- `classifier-service`:

  - `POST /predict` –
    - Input: `{ "features": [0.1, 0.5, 0.3, ...] }`
    - Output: `{ "label": "4", "probability": 0.87 }`

---

## 8. Demo Runbook (Reproducible Steps)

This section summarises the concrete steps to reproduce the demo on a fresh machine. All commands are intended to be run **from the project root**.

### 8.1 Prerequisites

- Linux or macOS host with:
  - Python 3.10/3.11 and `pip`
  - Docker
  - `minikube` 
  - `kubectl`
  - `jq` (for pretty-printing JSON; optional but handy)
- Sufficient resources (the examples below assume roughly `--cpus=2 --memory=16000` for minikube).

### 8.2 Start minikube (once per session)

```bash
minikube start --cpus=2 --memory=16000
minikube addons enable metrics-server
```

Verify:

```bash
kubectl get nodes
kubectl get pods -A
```

### 8.3 Train and register classifier models

From the project root:

```bash
# Activate your virtualenv / conda env if you use one
python services/classifier-service/train_mnist_rf.py \
  --n-estimators 100 \
  --max-depth 10 \
  --version-id mnist_rf_v1

python services/classifier-service/train_mnist_rf.py \
  --n-estimators 200 \
  --max-depth 12 \
  --version-id mnist_rf_v2
```

This populates the local "model registry" directory:

```bash
ls models/
# mnist_rf_v1.joblib
# mnist_rf_v2.joblib
```

The classifier service will later use an environment variable `MODEL_PATH` (e.g. `/models/mnist_rf_v1.joblib`) to select one of these artifacts at runtime.

### 8.4 Build Docker images for both services

From the project root:

```bash
# Classifier service image
docker build -t classifier-service:v1 services/classifier-service

# LM service image (GPT-style causal LM)
docker build -t lm-service:v1 services/lm-service
```

### 8.5 Load images into minikube

Minikube runs its own Docker daemon inside the VM/container. Load the freshly built images into that environment:

```bash
minikube image load classifier-service:v1
minikube image load lm-service:v1
```

### 8.6 Mount the models directory into minikube

The classifier pods expect model artifacts under `/models` inside the node. Use `minikube mount` to bind the host `models/` directory into the node filesystem:

```bash
# In a dedicated terminal (keep it running while the demo is active)
minikube mount "$(pwd)/models:/models"
```

This effectively turns the host `models/` directory into a very simple local model registry for the classifier deployment.

### 8.7 Deploy classifier and LM services to Kubernetes

Apply manifests for both services:

```bash
# Classifier: Deployment, Service, HPA
kubectl apply -f k8s/classifier/deployment.yaml
kubectl apply -f k8s/classifier/service.yaml
kubectl apply -f k8s/classifier/hpa.yaml

# LM: Deployment, Service, HPA
kubectl apply -f k8s/lm/deployment.yaml
kubectl apply -f k8s/lm/service.yaml
kubectl apply -f k8s/lm/hpa.yaml
```

Check that pods are up:

```bash
kubectl get deployments
kubectl get pods
kubectl get svc
kubectl get hpa
```

At this point you should see one pod for each deployment and two ClusterIP services.

### 8.8 Test both services

#### 8.8.1 Classifier service

Port-forward the classifier service to localhost:

```bash
# Terminal A
kubectl port-forward svc/classifier-service 8000:8000
```

In another terminal:

```bash
# Health check
curl http://127.0.0.1:8000/healthz | jq

# Simple prediction using the query client
python clients/query_classifier.py --sample-index 2001

# Basic metrics
curl http://127.0.0.1:8000/metrics-simple | jq
```

You should see a valid health response, a prediction with top-3 probabilities, and simple latency counters.

#### 8.8.2 LM service

Port-forward the LM service to localhost:

```bash
# Terminal B
kubectl port-forward svc/lm-service 8100:8000
```

In another terminal:

```bash
# Health check
curl http://127.0.0.1:8100/healthz | jq

# Single generation
python - << 'EOF'
import requests
payload = {"prompt": "Hello, I'm a language model,", "max_new_tokens": 40}
resp = requests.post("http://127.0.0.1:8100/generate", json=payload, timeout=20.0)
print(resp.status_code)
print(resp.json())
EOF
```

You should see a `200` response and a short generated continuation of the prompt.

### 8.9 Autoscaling demos

#### 8.9.1 Classifier autoscaling demo

The goal here is to drive enough load through `classifier-service` to trigger CPU-based autoscaling of the `classifier-deployment` while keeping the behaviour understandable.

1. **Ensure artificial CPU load is configured** in `k8s/classifier/deployment.yaml` (inside the container `env` section):

   ```yaml
   env:
     - name: MODEL_PATH
       value: /models/mnist_rf_v1.joblib
     - name: ARTIFICIAL_CPU_SEC
       value: "0.1"   # ~100 ms of extra CPU per request.
   ```

   Apply and restart the deployment:

   ```bash
   kubectl apply -f k8s/classifier/deployment.yaml
   kubectl rollout restart deployment classifier-deployment
   kubectl get pods
   ```

2. **Port-forward the classifier service** (if not already running):

   ```bash
   # Terminal A
   kubectl port-forward svc/classifier-service 8000:8000
   ```

3. **Start the load generator** in another terminal:

   ```bash
   python clients/load_test_classifier.py \
     --url http://127.0.0.1:8000/predict \
     --duration-seconds 120 \
     --concurrency 40 \
     --dataset-size 2000
   ```

   This spawns 40 worker threads, each repeatedly sending MNIST-derived feature vectors to `/predict` for two minutes.

4. **Watch the autoscaler and pod metrics** in parallel:

   ```bash
   # Terminal C: HPA behaviour
   kubectl get hpa -w

   # Terminal D: pod metrics (CPU/memory)
   kubectl top pods
   ```

Expected behaviour:

- Initially, a single classifier pod handles all traffic; CPU rises well above the 70% target.
- `classifier-hpa` transitions from `cpu: 0%/70%` to values like `cpu: 150%/70%`.
- The HPA scales `classifier-deployment` from 1 replica up to 2–3 (depending on configuration).
- After the load test finishes, CPU utilization drops and the HPA eventually scales back down to a single replica.

This demonstrates that a lightweight model service can be scaled horizontally to cope with increased traffic, and that the application includes explicit knobs (`ARTIFICIAL_CPU_SEC`, load generator concurrency) to make the scaling behaviour observable on small hardware.

#### 8.9.2 LM autoscaling demo

The goal here is to show that the LM service also autosscales, but with very different characteristics: lower RPS, heavier per-request cost, and (likely) more pronounced cold-start effects.

1. **Configure LM artificial CPU load (optional)** in `k8s/lm/deployment.yaml`:

   ```yaml
   env:
     - name: MODEL_NAME
       value: "erwanf/gpt2-mini"
     - name: MAX_TOTAL_TOKENS
       value: "256"
     - name: ARTIFICIAL_CPU_SEC
       value: "0.0"   # start with 0.0; the model itself is already heavy. Only add non-zero value if scaling does not work
   ```

   Apply and restart:

   ```bash
   kubectl apply -f k8s/lm/deployment.yaml
   kubectl rollout restart deployment lm-deployment
   kubectl get pods
   ```

2. **Port-forward the LM service** (if not already running):

   ```bash
   # Terminal B
   kubectl port-forward svc/lm-service 8100:8000
   ```

3. **Optionally warm up the model** with a few manual requests to reduce early timeouts:

   ```bash
   python - << 'EOF'
   import requests
   for i in range(3):
       r = requests.post(
           "http://127.0.0.1:8100/generate",
           json={"prompt": "Warmup run", "max_new_tokens": 40},
           timeout=30.0,
       )
       print(i, r.status_code)
   EOF
   ```

4. **Run the LM load generator** with modest concurrency:

   ```bash
   python clients/load_test_lm.py \
     --url http://127.0.0.1:8100/generate \
     --duration-seconds 120 \
     --concurrency 2 \
     --max-new-tokens 40
   ```

   Higher concurrency (e.g. 5–10) is possible but on a 2 vCPU minikube node it quickly leads to request timeouts, which is interesting to observe but less visually clean for a short demo.

5. **Observe HPA and pod metrics** as before:

   ```bash
   kubectl get hpa -w
   kubectl top pods
   ```

Expected behaviour:

- Even at low RPS (around 1 req/s), each `/generate` call is expensive enough on CPU that LM pod CPU utilization climbs above the 70% target.
- `lm-hpa` scales the `lm-deployment` from 1 to up to 3 replicas.
- New replicas must also load the model, leading to noticeable cold-start effects (longer latency, potential initial timeouts) before stabilising.
- After the load test, utilization drops and the HPA eventually scales back down to a single replica.

The contrast with the classifier service is intentional and illustrates how different model profiles (lightweight sklearn vs heavy LM) interact with the same autoscaling mechanisms.

#### 8.9.3 Sample output

On my local machine, I get the following scaling behavior when running both autoscaling demos together:

```
> kubectl get hpa -w
NAME             REFERENCE                          TARGETS       MINPODS   MAXPODS   REPLICAS   AGE
classifier-hpa   Deployment/classifier-deployment   cpu: 0%/70%   1         5         1          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 0%/70%   1         3         1          4d
lm-hpa           Deployment/lm-deployment           cpu: 185%/70%   1         3         1          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 162%/70%   1         5         1          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 185%/70%   1         3         3          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 162%/70%   1         5         3          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 199%/70%   1         3         3          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 200%/70%   1         5         3          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 7%/70%     1         3         3          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 13%/70%    1         5         3          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 0%/70%     1         3         3          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 0%/70%     1         5         3          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 0%/70%     1         3         3          4d
lm-hpa           Deployment/lm-deployment           cpu: 0%/70%     1         3         3          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 0%/70%     1         5         3          4d5h
lm-hpa           Deployment/lm-deployment           cpu: 0%/70%     1         3         1          4d
classifier-hpa   Deployment/classifier-deployment   cpu: 0%/70%     1         5         1          4d5h
```

---

### 8.10 Cleanup

To clean the resources created during the demo:

```bash
# Delete K8s resources
kubectl delete -f k8s/classifier/hpa.yaml
kubectl delete -f k8s/classifier/service.yaml
kubectl delete -f k8s/classifier/deployment.yaml

kubectl delete -f k8s/lm/hpa.yaml
kubectl delete -f k8s/lm/service.yaml
kubectl delete -f k8s/lm/deployment.yaml

# (Optional) stop minikube
minikube stop
```

The trained model artifacts under `models/` are kept, acting as a simple local model registry that can be reused in future runs or extended towards a more formal registry-based workflow.

To tear down the resources created during the demo:

```bash
# Delete K8s resources
kubectl delete -f k8s/classifier/hpa.yaml
kubectl delete -f k8s/classifier/service.yaml
kubectl delete -f k8s/classifier/deployment.yaml

kubectl delete -f k8s/lm/hpa.yaml
kubectl delete -f k8s/lm/service.yaml
kubectl delete -f k8s/lm/deployment.yaml

# (Optional) stop minikube
minikube stop
```

The trained model artifacts under `models/` are kept, acting as a simple local model registry that can be reused in future runs.
