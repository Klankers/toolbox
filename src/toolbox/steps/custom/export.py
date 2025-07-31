"""Class definition for exporting data steps."""

#### Mandatory imports ####
from ..base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag


@register_step
class ExportStep(BaseStep):
    step_name = "Data Export"

    def run(self):
        self.log(
            f"Exporting data in {self.parameters['export_format']} format to {self.parameters['output_path']}"
        )

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        else:
            self.log(f"Data found in context.")
        data = self.context["data"]
        export_format = self.parameters["export_format"]
        output_path = self.parameters["output_path"]

        # Validate the export format
        if export_format not in ["csv", "netcdf", "hdf5", "parquet"]:
            raise ValueError(
                f"Unsupported export format: {export_format}. Supported formats are: csv, netcdf, hdf5, parquet."
            )
        if not output_path:
            raise ValueError("Output path must be specified for data export.")
        # Ensure the output path is a string
        if not isinstance(output_path, str):
            raise ValueError("Output path must be a string.")

        # Export data based on the specified format
        if export_format == "csv":
            data.to_csv(output_path)
        elif export_format == "netcdf":
            data.to_netcdf(output_path, engine="netcdf4")
        elif export_format == "hdf5":
            data.to_netcdf(output_path, engine="h5netcdf")
        elif export_format == "parquet":
            data.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        self.log(f"Data exported successfully to {output_path}")
        return self.context

    def generate_diagnostics(self):
        """Generate diagnostics for the export step."""
        self.log(f"Generating diagnostics for {self.step_name}")
        diag.generate_diagnostics(self.context, self.step_name)
        self.log(f"Diagnostics generated successfully.")
