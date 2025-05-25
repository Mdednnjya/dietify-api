# src/utils/mlflow_manager.py
import mlflow
import logging
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLflowManager:
    """MLflow manager dengan graceful fallback untuk production"""

    def __init__(self, tracking_uri: str = None, enabled: bool = True):
        self.enabled = enabled and os.getenv('MLFLOW_ENABLED', 'true').lower() == 'true'
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.connected = False
        self.current_run = None

        if self.enabled:
            self._setup_connection()

    def _setup_connection(self):
        """Setup MLflow connection dengan error handling"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Test connection dengan timeout singkat
            import requests
            response = requests.get(f"{self.tracking_uri}/health", timeout=2)
            if response.status_code == 200:
                self.connected = True
                logger.info(f"✅ MLflow connected: {self.tracking_uri}")
            else:
                logger.warning(f"⚠️ MLflow unhealthy: {response.status_code}")

        except Exception as e:
            logger.warning(f"⚠️ MLflow unavailable: {e}")
            self.connected = False

    @contextmanager
    def start_run(self, run_name: str = None, experiment_name: str = None, nested: bool = False):
        """Context manager untuk MLflow runs dengan fallback"""
        if not self.enabled or not self.connected:
            # Fallback mode - return dummy context
            yield DummyRun()
            return

        try:
            # Set experiment jika diperlukan
            if experiment_name:
                try:
                    mlflow.set_experiment(experiment_name)
                except Exception as e:
                    logger.warning(f"Failed to set experiment {experiment_name}: {e}")

            # Start run
            run = mlflow.start_run(run_name=run_name, nested=nested)
            self.current_run = run

            try:
                yield MLflowRun(run)
            finally:
                mlflow.end_run()
                self.current_run = None

        except Exception as e:
            logger.error(f"MLflow run error: {e}")
            # Fallback ke dummy run jika ada error
            yield DummyRun()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters dengan error handling"""
        if not self.enabled or not self.connected or not self.current_run:
            return

        try:
            # Convert semua values ke string untuk MLflow
            safe_params = {}
            for key, value in params.items():
                if value is not None:
                    safe_params[key] = str(value)

            mlflow.log_params(safe_params)

        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics dengan error handling"""
        if not self.enabled or not self.connected or not self.current_run:
            return

        try:
            # Filter hanya numeric values
            safe_metrics = {}
            for key, value in metrics.items():
                try:
                    safe_metrics[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-numeric metric {key}: {value}")

            if safe_metrics:
                mlflow.log_metrics(safe_metrics, step=step)

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact dengan error handling"""
        if not self.enabled or not self.connected or not self.current_run:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def set_tag(self, key: str, value: str):
        """Set tag dengan error handling"""
        if not self.enabled or not self.connected or not self.current_run:
            return

        try:
            mlflow.set_tag(key, str(value))
        except Exception as e:
            logger.warning(f"Failed to set tag {key}: {e}")


class MLflowRun:
    """Wrapper untuk MLflow run yang aktif"""

    def __init__(self, run):
        self.run = run
        self.info = run.info if run else None

    @property
    def run_id(self):
        return self.info.run_id if self.info else "no-run"


class DummyRun:
    """Dummy run untuk fallback mode"""

    def __init__(self):
        self.info = None
        self.run_id = "fallback-mode"


# Global MLflow manager instance
mlflow_manager = MLflowManager()


# Convenience functions
def start_run(run_name: str = None, experiment_name: str = None, nested: bool = False):
    """Start MLflow run dengan fallback"""
    return mlflow_manager.start_run(run_name, experiment_name, nested)


def log_params(params: Dict[str, Any]):
    """Log parameters dengan fallback"""
    mlflow_manager.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int = None):
    """Log metrics dengan fallback"""
    mlflow_manager.log_metrics(metrics, step)


def log_artifact(local_path: str, artifact_path: str = None):
    """Log artifact dengan fallback"""
    mlflow_manager.log_artifact(local_path, artifact_path)


def set_tag(key: str, value: str):
    """Set tag dengan fallback"""
    mlflow_manager.set_tag(key, value)