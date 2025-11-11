"""Class definition for quality control steps."""

#### Mandatory imports ####
from ..base_step import BaseStep, register_step
import toolbox.utils.diagnostics as diag
from toolbox.steps import QC_CLASSES

#### Custom imports ####
import polars as pl


@register_step
class ApplyQC(BaseStep):
    
    step_name = "Apply QC"

    def organise_flags(self, new_flags):
        # Method for taking in new flags and cross checking against exiting flags, including upgrading flags when necessary.
        # Update existing flag columns
        flag_columns_to_update = set(new_flags.columns) & set(self.flag_store.columns)
        for column_name in flag_columns_to_update:
            self.flag_store = self.flag_store.with_columns(
                pl.max_horizontal([pl.col(column_name), new_flags[column_name]]).alias(
                    f"{column_name}"
                )
            )
        # Add new QC flag columns if they dont already exist
        flag_columns_to_add = set(new_flags.columns) - set(self.flag_store.columns)
        if len(flag_columns_to_add) > 0:
            self.flag_store = self.flag_store.with_columns(
                new_flags[list(flag_columns_to_add)],
            )

    def run(self):

        # Defining the order of operations
        if len(self.qc_settings.keys()) == 0:
            raise KeyError("[Apply QC] No QC operations were specified in an ApplyQC step.")
        else:
            invalid_requests = set(self.qc_settings.keys()) - set(QC_CLASSES.keys())
            if invalid_requests:
                raise KeyError(f"[Apply QC] The following requested QC tests could not be found: {invalid_requests}")
        queued_qc = [QC_CLASSES.get(key) for key in self.qc_settings.keys()]

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("[Apply QC] No data found in context. Please load data first.")
        else:
            self.log("Data found in context.")
        data = self.context["data"].copy()

        # Try and fetch the qc history from context and update it
        qc_history = self.context.setdefault("qc_history", {})
        
        # Collect all of the required varible names and qc outputs
        all_required_variables = set({})
        test_qc_outputs_cols = set({})
        for test in queued_qc:
            all_required_variables.update(test.required_variables)
            test_qc_outputs_cols.update(test.qc_outputs)

        # Convert data to polars for fast processing
        if not set(all_required_variables).issubset(set(data.keys())):
            raise KeyError(
                f"[Apply QC] The data is missing variables: ({set(all_required_variables) - set(data.keys())}) which are required for QC."
                f" Make sure that the variables are present in the data, or use remove tests from the order."
            )

        # Fetch existing flags from the data and create a place to store them
        existing_flags = [
            flag_col for flag_col in data.data_vars if flag_col in test_qc_outputs_cols
        ]
        self.flag_store = pl.DataFrame()
        if len(existing_flags) > 0:
            self.flag_store = pl.from_pandas(
                data[existing_flags].to_dataframe(), nan_to_null=False
            )

        # Run through all of the QC steps and add the flags to flag_store
        for qc_test_name, qc_test_params in self.qc_settings.items():
            # Create an instance of this test step
            print(f"[Apply QC] Applying: {qc_test_name}")
            qc_test_instance = QC_CLASSES[qc_test_name](data, **qc_test_params)
            returned_flags = qc_test_instance.return_qc()
            self.organise_flags(returned_flags)

            # Update QC history
            for flagged_var in returned_flags.columns:
                percent_flagged = (returned_flags[flagged_var].to_numpy() != 0).sum() / len(returned_flags)
                qc_history.setdefault(flagged_var, []).append((qc_test_name, percent_flagged))

            # Diagnostic plotting
            if self.diagnostics:
                qc_test_instance.plot_diagnostics()
                
            # Once finished, remove the test instance from memory
            del qc_test_instance

        # Append the flags from self.flag_store to the xarray data and push back into context
        for flag_column in self.flag_store.columns:
            data[flag_column] = (
                ("N_MEASUREMENTS",),
                self.flag_store[flag_column].to_numpy(),
            )
        self.context["data"] = data
        self.context["qc_history"] = qc_history
        return self.context