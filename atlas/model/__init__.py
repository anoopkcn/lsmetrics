import importlib
import inspect
from typing import Type
from .csgann import CSGANN
from .csgcnn import CSGCNN
from .csgcnn_vae import CSGCNN_VAE
from .flowgnn import FlowGNN
from .flow import GNFlow


def get_model(model_name: str) -> Type:
    for module_name in [
        "csgnn.model",
        "csgnn.model.custom",
    ]:  # Add more module paths if needed
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.lower() == model_name.lower():
                    return obj
        except ImportError:
            continue

    raise ValueError(f"Model {model_name} not found")


def get_available_models():
    models = []
    for module_name in [
        "csgnn.model",
        "csgnn.model.custom",
    ]:  # Add more module paths if needed
        try:
            module = importlib.import_module(module_name)
            models.extend(
                [
                    name
                    for name, obj in inspect.getmembers(module)
                    # if inspect.isclass(obj)
                    # and issubclass(obj, (CSGANN, CSGCNN, CSGCNN_VAE, FlowGNN))
                ]
            )
        except ImportError:
            continue
    return list(set(models))  # Remove duplicates
