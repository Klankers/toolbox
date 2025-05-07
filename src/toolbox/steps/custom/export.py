"""Class definition for exporting data steps."""

#### Mandatory imports ####
from ..base_step import BaseStep
import utils.diagnostics as diag


class ExportStep(BaseStep):
    step_name = "Data Export"

    def run(self):
        print(
            f"[Export] Exporting data in {self.parameters['export_format']} format to {self.parameters['output_path']}"
        )

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            print(f"[Export] Data found in context.")
        data = self.context["data"]
        export_format = self.parameters["export_format"]
        output_path = self.parameters["output_path"]
        # Export data based on the specified format
        if export_format == "csv":
            data.to_csv(output_path)
        elif export_format == "netcdf":
            data.to_netcdf(output_path, engine="netcdf4")
        elif export_format == "parquet":
            data.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        print(f"[Export] Data exported successfully to {output_path}")
        return self.context

    def generate_diagnostics(self):
        """Generate diagnostics for the export step."""
        print(f"[Export] Generating diagnostics for {self.step_name}")
        diag.generate_diagnostics(self.context, self.step_name)
        print(f"[Export] Diagnostics generated successfully.")
