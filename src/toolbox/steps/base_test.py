

# Registry of explicitly registered step classes
REGISTERED_QC = {}
flag_cols = {
    0: "gray",
    1: "blue",
    2: "lightblue",
    3: "orange",
    4: "red",
    5: "gray",
    6: "gray",
    7: "gray",
    8: "cyan",
    9: "black"
}


def register_qc(cls):
    """Decorator to mark QC tests that can be accessed by the ApplyQC step."""
    test_name = getattr(cls, "test_name", None)
    if test_name is None:
        raise ValueError(
            f"QC test {cls.__name__} is missing required 'test_name' attribute."
        )
    REGISTERED_QC[test_name] = cls
    return cls


class BaseTest:
    """
    Target Variable:
    Flag Number:
    Variables Flagged:
    ? description ?
    """

    test_name = None
    expected_parameters = {}
    required_variables = []
    qc_outputs = []

    def __init__(self, data, **kwargs):
        self.data = data.copy()

        invalid_params = set(kwargs.keys()) - set(self.expected_parameters.keys())
        if invalid_params:
            raise KeyError(f"Unexpected parameters for {self.test_name}: {invalid_params}")

        for k, v in kwargs.items():
            self.expected_parameters[k] = v

        for k, v in self.expected_parameters.items():
            setattr(self, k, v)

        self.flags = None

    def return_qc(self):
        self.flags = None  # replace with processing of some kind
        return self.flags

    def plot_diagnostics(self):
        # Any relevant diagnostic
        pass
