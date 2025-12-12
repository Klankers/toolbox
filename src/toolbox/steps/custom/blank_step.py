#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

#### Custom imports ####


@register_step
class BlankStep(BaseStep, QCHandlingMixin):

    step_name = "Blank Step"

    def run(self):
        self.filter_qc()

        # EXAMPLE: self.data["C"] = self.data["A"] * self.data["B"]

        self.reconstruct_data()
        self.update_qc()

        # If a new variable was added, use self.generate_qc()
        # EXAMPLE: self.generate_qc({"C_QC": ["A_QC", "B_QC"]})

        if self.diagnostics:
            self.generate_diagnostics()

        self.context["data"] = self.data
        return self.context

    def generate_diagnostics(self):
        pass
