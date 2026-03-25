"""
Load test for NeuroFlow API.
Run: locust -f tests/performance/locustfile.py -H http://localhost:8000 --headless -u 50 -r 5 --run-time 5m
"""
import random
import json
from locust import HttpUser, task, between, events

SAMPLE_QUERIES = [
    "What is retrieval augmented generation?",
    "How does HNSW indexing work?",
    "Explain the attention mechanism in transformers",
    "What are the main components of a RAG pipeline?",
    "How do embeddings represent semantic meaning?",
    "What is the difference between dense and sparse retrieval?",
    "Explain reciprocal rank fusion",
    "How does cross-encoder reranking improve results?",
]

# Will be populated after auth
AUTH_TOKEN = None


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global AUTH_TOKEN
    response = environment.runner.environment.host
    try:
        import requests
        resp = requests.post(
            f"{response}/auth/token",
            json={"client_id": "admin-client", "client_secret": "admin-secret-change-in-prod"},
            timeout=10,
        )
        if resp.status_code == 200:
            AUTH_TOKEN = resp.json().get("access_token")
    except Exception:
        pass


class QueryUser(HttpUser):
    weight = 7
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
        # Get a pipeline to query
        resp = self.client.get("/api/v1/pipelines", headers=self.headers)
        self.pipeline_id = None
        if resp.status_code == 200:
            pipelines = resp.json()
            if pipelines:
                self.pipeline_id = pipelines[0]["id"]

    @task
    def query_pipeline(self):
        if not self.pipeline_id:
            return
        self.client.post(
            "/api/v1/query",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "pipeline_id": self.pipeline_id,
                "stream": False,
            },
            headers=self.headers,
            name="/api/v1/query",
        )

    @task(3)
    def check_health(self):
        self.client.get("/health", name="/health")


class IngestUser(HttpUser):
    weight = 2
    wait_time = between(5, 15)

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

    @task
    def check_documents(self):
        self.client.get("/api/v1/pipelines", headers=self.headers, name="/api/v1/pipelines")


class AdminUser(HttpUser):
    weight = 1
    wait_time = between(3, 8)

    def on_start(self):
        self.headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}

    @task
    def check_evaluations(self):
        self.client.get("/api/v1/evaluations", headers=self.headers, name="/api/v1/evaluations")

    @task
    def check_aggregate(self):
        self.client.get("/api/v1/evaluations/aggregate", headers=self.headers, name="/api/v1/evaluations/aggregate")
