# Registry of explicitly registered step classes
REGISTERED_STEPS = {}


def register_step(cls):
    """Decorator to mark a step class for inclusion in the pipeline."""
    step_name = getattr(cls, "step_name", None)
    if step_name is None:
        raise ValueError(
            f"Class {cls.__name__} is missing required 'step_name' attribute."
        )
    REGISTERED_STEPS[step_name] = cls
    return cls


class BaseStep:
    def __init__(self, name, parameters=None, diagnostics=False, context=None):
        self.name = name
        self.parameters = parameters or {}
        self.diagnostics = diagnostics
        self.context = context or {}

        # add attrs from parameters to self
        for key, value in self.parameters.items():
            setattr(self, key, value)

    def run(self):
        """To be implemented by subclasses"""
        raise NotImplementedError(f"Step '{self.name}' must implement a run() method.")
        return self.context

    def generate_diagnostics(self):
        """Hook for diagnostics (optional)"""
        pass

    def log(self, message):
        """Log messages with step name"""
        print(f"[{self.name}] {message}")
