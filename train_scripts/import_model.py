import importlib
import inspect

def import_model(model_name, **kwargs):
    try:
        device = kwargs["device"]

        # import the model's class and check it's init parameters
        module = importlib.import_module(f"models.{model_name}")
        model_class = getattr(module, model_name)
        sig = inspect.signature(model_class.__init__)
        
        # filter the model's parameters inside the kwargs
        valid_params = {name for name, _ in sig.parameters.items() if name != 'self'}

        # Filter provided kwargs down to only those valid for this class
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        model = model_class(**filtered_kwargs)
        return model.to(device)

    except (ModuleNotFoundError, AttributeError, TypeError) as e:
        raise ValueError(f"Failed to import or initialize model '{model_name}': {e}")
