"""Class definition for Calibration steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag


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
