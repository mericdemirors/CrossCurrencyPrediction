import importlib

def import_loss(loss_names):
    loss_functions = []
    for name in loss_names:
        try:
            module = importlib.import_module(f'additional_losses.{name}')
            loss_fn = getattr(module, name)
            loss_functions.append(loss_fn)
        except (ModuleNotFoundError, AttributeError) as e:
            print(f'Error importing "{name}": {e}')
    
    return loss_functions
