import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class MLflowManager:
    """Production-ready MLflow integration for FastAPI"""

    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.enabled = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
        self.client = None
        self.connected = False

        if self.enabled:
            self._initialize_connection()

    def _initialize_connection(self):
        """Initialize MLflow connection with retry logic"""
        max_retries = 10
        retry_delay = 3

        logger.info(f"Initializing MLflow connection to {self.tracking_uri}")

        for attempt in range(max_retries):
            try:
                # Force set tracking URI globally
                mlflow.set_tracking_uri(self.tracking_uri)
                self.client = MlflowClient(self.tracking_uri)

                # Test connection with timeout
                self.client.search_experiments(max_results=1)
                self.connected = True
                logger.info(f"✅ MLflow connected successfully to {self.tracking_uri}")
                return

            except Exception as e:
                logger.warning(f"MLflow connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed to connect to MLflow after all retries. URI: {self.tracking_uri}")
                    self.connected = False

    def is_available(self) -> bool:
        """Check if MLflow is available"""
        return self.enabled and self.connected

    @contextmanager
    def start_run(self, experiment_name: str, run_name: str = None):
        """Context manager for MLflow runs with error handling"""
        if not self.is_available():
            logger.info("MLflow not available, skipping tracking")
            yield None
            return

        try:
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Could not set experiment {experiment_name}: {e}")
                experiment_id = None

            # Start run
            with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=run_name
            ) as run:
                yield run

        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            yield None

    def log_params_safe(self, params: dict):
        """Safe parameter logging"""
        if not self.is_available():
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_metrics_safe(self, metrics: dict):
        """Safe metrics logging"""
        if not self.is_available():
            return

        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifact_safe(self, local_path: str, artifact_path: str = None):
        """Safe artifact logging"""
        if not self.is_available():
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def health_check(self) -> dict:
        """Health check for MLflow connection with SQLite backend"""
        if not self.enabled:
            return {"status": "disabled", "message": "MLflow tracking disabled"}

        if not self.connected:
            return {"status": "disconnected", "message": "MLflow server not available"}

        try:
            # Test basic operations
            experiments = self.client.search_experiments(max_results=1)
            return {
                "status": "connected",
                "backend": "sqlite",
                "tracking_uri": self.tracking_uri,
                "experiments_count": len(experiments)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global MLflow manager instance
mlflow_manager = MLflowManager()