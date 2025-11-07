"""Class definition for deriving CTD variables."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####
import polars as pl
import numpy as np
import gsw


@register_step
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Blank Step"

    def run(self):
        self.filter_qc()

        # self.data["B"] = self.data["B"] * 2
        self.data["C"] = self.data["A"] * self.data["B"]

        self.reconstruct_data()
        self.update_qc()
        self.generate_qc({"C_QC": ["A_QC", "B_QC"]})

        self.generate_diagnostics()
        return self.context

    def generate_diagnostics(self):
        self.print_qc_settings()
