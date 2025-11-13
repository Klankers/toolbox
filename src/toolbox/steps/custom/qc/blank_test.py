#### Mandatory imports ####
from toolbox.steps.base_test import BaseTest, register_qc, flag_cols

#### Custom imports ####


@register_qc
class impossible_date_test(BaseTest):
    """
    Target Variable: TIME
    Flag Number: 4 (bad data)
    Variables Flagged: TIME
    Checks that the datetime of each point is valid.
    """

    test_name = ""
    expected_parameters = {}
    required_variables = []
    qc_outputs = []

    def return_qc(self):
        # Access the data with self.data
        # self.flags should be an xarray Dataset with data_vars that hold the "{variable}_QC" columns produced by the test
        return self.flags

    def plot_diagnostics(self):
        plt.show(block=True)