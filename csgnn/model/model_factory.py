import importlib


def get_model(model_name):
    try:
        module = importlib.import_module(f"csgnn.model.{model_name.lower()}")
        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError):
        raise ValueError(f"Model {model_name} not found")


def get_available_models():
    # This list should be updated when new models are added
    return ["CSGCNN", "CSGANN"]
