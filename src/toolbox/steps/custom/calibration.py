"""Class definition for Calibration steps."""

#### Mandatory imports ####
from toolbox.steps.base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag


class FactoryCalibrationStep(BaseStep):
    step_name = "Factory Calibration"

    def run(self):
        print(f"[FactoryCalibration] Applying calibration")
        return self.context


class SensorCalibrationStep(BaseStep):
    step_name = "Sensor Calibration"

    def run(self):
        print(f"[SensorCalibration] Calibrating sensor")
        return self.context
