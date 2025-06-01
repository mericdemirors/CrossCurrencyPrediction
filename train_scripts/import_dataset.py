import importlib
import inspect

def import_dataset(dataset_name, **kwargs):
    try:
        # import the dataset's class and check it's init parameters
        module = importlib.import_module(f"dataset_classes.{dataset_name}")
        dataset_class = getattr(module, dataset_name)
        sig = inspect.signature(dataset_class.__init__)
        
        # filter the dataset's parameters inside the kwargs
        valid_params = {name for name, _ in sig.parameters.items() if name != 'self'}

        # Filter provided kwargs down to only those valid for this class
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        dataset = dataset_class(**filtered_kwargs)
        return dataset

    except (ModuleNotFoundError, AttributeError, TypeError) as e:
        raise ValueError(f"Failed to import or initialize dataset '{dataset_name}': {e}")
