"""Pipeline Class"""

import yaml
import pandas as pd
import numpy as np
import xarray as xr
from toolbox.utils.diagnostics import (
    summarising_profiles,
    find_closest_prof,
    plot_distance_time_grid,
    plot_glider_pair_heatmap_grid,
)
from toolbox.utils.alignment import (
    interpolate_DEPTH,
    aggregate_vars,
    merge_depth_binned_profiles,
    compute_r2_for_merged_profiles,
)

from toolbox.steps import create_step, STEP_CLASSES, STEP_DEPENDENCIES
from graphviz import Digraph


class Pipeline:
    def __init__(self, config_path=None):
        """Initialize pipeline with optional config file"""
        self.steps = []
        # Define the graph for visualiation
        self.graph = Digraph("Pipeline", format="png", graph_attr={"rankdir": "TB"})

        if config_path:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            self.global_parameters = config["pipeline"]
            self.build_steps(config["steps"])
        self._context = None

    def build_steps(self, steps_config, parent_name=None):
        """Recursively build steps from configuration"""
        for step in steps_config:
            # Check if the step has required stpes already imported
            REQUIRED_STEPS = STEP_DEPENDENCIES.get(step["name"], [])
            for required_step in REQUIRED_STEPS:
                if required_step not in STEP_CLASSES:
                    raise ValueError(
                        f"Required step '{required_step}' for '{step['name']}' is not found."
                    )
            self.add_step(
                step_name=step["name"],
                parameters=step.get("parameters", {}),
                diagnostics=step.get("diagnostics", False),
                parent_name=parent_name,
                run_immediately=False,
            )
            # Recurse into substeps
            if "substeps" in step:
                self.build_steps(step["substeps"], parent_name=step["name"])

    def add_step(
        self,
        step_name,
        parameters=None,
        diagnostics=False,
        parent_name=None,
        run_immediately=False,
    ):
        """Dynamically adds a step and optionally runs it immediately"""
        if step_name not in STEP_CLASSES:
            print(STEP_CLASSES)
            # Check if the step is recognised
            raise ValueError(
                f"Step '{step_name}' is not recognized. or missing @register_step."
            )

        step_config = {
            "name": step_name,
            "parameters": parameters or {},
            "diagnostics": diagnostics,
            "substeps": [],
        }

        if parent_name:
            # Add step as a substep of a parent if it exists
            parent = self._find_step(self.steps, parent_name)
            if parent:
                parent["substeps"].append(step_config)
            else:
                raise ValueError(f"Parent step '{parent_name}' not found.")
        else:
            self.steps.append(step_config)

        print(f"Step '{step_name}' added successfully!")

        if run_immediately:
            print(f"Running step '{step_name}' immediately.")
            self._context = self.execute_step(step_config, self._context)

    def _find_step(self, steps_list, step_name):
        """Recursively find a step by name"""
        for step in steps_list:
            if step["name"] == step_name:
                return step
            found = self._find_step(step.get("substeps", []), step_name)
            if found:
                return found
        return None

    def execute_step(self, step_config, _context):
        """Executes a single step"""
        step = create_step(step_config, _context)
        print(f"Executing: {step.name}")
        return step.run()

    def run_last_step(self):
        """Runs only the most recently added step"""
        if not self.steps:
            print("No steps to run.")
            return

        last_step = self.steps[-1]
        print(f"Running last step: {last_step['name']}")
        self.context = self.execute_step(last_step, self._context)

    def run(self):
        """Runs the entire pipeline"""

        for step in self.steps:
            self._context = self.execute_step(step, self._context)

        if self.global_parameters.get("visualisation", False):
            self.visualise_pipeline()

    def visualise_pipeline(self):
        """Generates a visualiation of the pipeline execution"""
        self.graph.clear()

        def add_to_graph(step_config, parent_name=None, step_order=None):
            step_name = step_config["name"]
            diagnostics = step_config.get("diagnostics", False)
            # ensure graphviz node colourway is clear that diagnostics are enabled
            color = "red" if diagnostics else "black"
            self.graph.node(
                step_name,
                step_name,
                color=color,
                style="filled",
                fillcolor="lightblue" if diagnostics else "white",
            )

            if parent_name:
                self.graph.edge(parent_name, step_name)

            # Add an edge to show the order/flow of substeps

            if step_order and len(step_order) > 1:
                for i in range(len(step_order) - 1):
                    self.graph.edge(step_order[i], step_order[i + 1])

            # Recursively add substeps

            substep_order = []
            for substep in step_config.get("substeps", []):
                substep_order.append(substep["name"])
                add_to_graph(substep, parent_name=step_name, step_order=substep_order)

        # Start by iterating through all top-level steps
        for step in self.steps:
            step_order = [step["name"]]  # Top-level step order
            add_to_graph(step, step_order=step_order)

        self.graph.render("pipeline_visualisation", view=True)

    def generate_config(self):
        """Generate a configuration dictionary from the current pipeline setup"""
        return {
            "pipeline": (
                self.global_parameters if hasattr(self, "global_parameters") else {}
            ),
            "steps": self.steps,
        }

    def export_config(self, output_path="generated_pipeline.yaml"):
        config_dict = self.generate_config()
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)
        print(f"Pipeline config exported to {output_path}")
        return config_dict


class PipelineManager:
    """A class enabling the execution of multiple pipelines in sequence."""

    def __init__(self):
        self.pipelines = {}  # {pipeline_name: Pipeline instance}
        self.alignment_map = {}  # {standard_name: {pipeline_name: alias}}
        self._contexts = None  # stores the result of get_contexts()
        self.settings = {}
        self._summary_ran = False

    def load_mission_control(self, config_path):
        """Load pipeline and alignment configuration from a mission control YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Load pipelines
        for entry in config.get("pipelines", []):
            self.add_pipeline(entry["name"], entry["config"])

        # Load alignment variable aliases
        alignment_vars = config.get("alignment", {}).get("variables", {})
        for std_var, details in alignment_vars.items():
            aliases = details.get("aliases", {})
            self.alignment_map[std_var] = aliases

        # Load settings
        self.settings = config.get("settings", {})

    def add_pipeline(self, name, config_path):
        """Add a single pipeline with a unique name."""
        if name in self.pipelines:
            raise ValueError(f"Pipeline '{name}' already added.")
        self.pipelines[name] = Pipeline(config_path)

    def run_all(self):
        """Run all registered pipelines and cache the resulting contexts."""
        for name, pipeline in self.pipelines.items():
            print("#" * 20)
            print(f"Running pipeline: {name}")
            pipeline.run()

        self._contexts = self.get_contexts()
        print("#" * 20)
        print("All pipelines executed successfully.")
        print(f"Contexts cached: {self._contexts.keys()}")
        print("#" * 20)

    def get_contexts(self):
        """Retrieve the context dictionary from each pipeline."""
        return {name: p._context for name, p in self.pipelines.items()}

    def get_cached_contexts(self):
        """Return previously cached contexts after run_all()."""
        if self._contexts is None:
            raise RuntimeError("Pipelines must be run before accessing contexts.")
        return self._contexts

    def summarise_all_profiles(self) -> pd.DataFrame:
        """
        For all pipelines, summarise profiles and plot glider-to-glider distance time series.
        This includes:
            - Computing median TIME, LATITUDE, LONGITUDE per profile
            - Matching each profile to its closest in time from another source
            - Plotting a distance grid comparing all gliders

        Returns
        -------
        pd.DataFrame
            Concatenated summary of all glider profiles, with closest match info appended.
        """
        self._summary_ran = True
        if self._contexts is None:
            raise RuntimeError("Pipelines must be run before generating summaries.")

        print("[Pipeline Manager] Generating glider distance summaries...")
        # Step 1: Generate per-glider summaries
        self.summary_per_glider = {}
        for pipeline_name, context in self._contexts.items():
            ds = context["data"]
            if not isinstance(ds, xr.Dataset):
                raise TypeError(f"Pipeline '{pipeline_name}' has invalid dataset.")
            else:
                print(f"[Pipeline Manager] Processing dataset for {pipeline_name}...")

            summary_df = summarising_profiles(ds, pipeline_name)
            print("Summary Columns:", summary_df.columns.tolist())
            self.summary_per_glider[pipeline_name] = summary_df

        # Step 2: Find closest profiles across gliders
        # Extract diagnostic flags from settings
        show_plots = self.settings.get("diagnostics", {}).get("show_plots", True)
        save_plots = self.settings.get("diagnostics", {}).get("save_plots", False)
        distance_over_time_matrix = self.settings.get("diagnostics", {}).get(
            "distance_over_time_matrix", False
        )
        self.matchup_thresholds = self.settings.get("diagnostics", {}).get(
            "matchup_thresholds", {}
        )
        max_time_threshold = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_time_threshold", 12)
        )
        max_distance_threshold = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_distance_threshold", 20)
        )
        bin_size = (
            self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("bin_size", 2)
        )
        if show_plots:
            if not distance_over_time_matrix:
                print("[Pipeline Manager] Distance over time matrix is disabled.")
            else:
                print("[Pipeline Manager] Plotting distance time grid...")
                # After generating all summaries...
                combined_summaries = plot_distance_time_grid(
                    summaries=self.summary_per_glider,
                    output_path=self.settings.get("diagnostics", {}).get(
                        "distance_plot_output", None
                    ),
                    show=self.settings.get("diagnostics", {}).get("show_plots", True),
                )

            if not self.matchup_thresholds:
                print(
                    "[Pipeline Manager] Matchup thresholds are not set. Skipping heatmap grid."
                )
            else:
                print("[Pipeline Manager] Finding closest profiles across gliders...")
                # compute time taken for caluclations
                start_time = pd.Timestamp.now()
                plot_glider_pair_heatmap_grid(
                    summaries=self.summary_per_glider,
                    time_bins=np.arange(0, max_time_threshold + 1, bin_size),
                    dist_bins=np.arange(0, max_distance_threshold + 1, bin_size),
                    output_path=self.settings.get("diagnostics", {}).get(
                        "heatmap_output", None
                    ),
                    show=self.settings.get("diagnostics", {}).get("show_plots", True),
                )
                end_time = pd.Timestamp.now()
                print(
                    f"[Pipeline Manager] Heatmap grid plotted in {end_time - start_time}"
                )

        else:
            print("[Pipeline Manager] Skipping plots.")
            # Generate combined_summaries without plotting
            combined_summaries = []
            for i, g_id in self.summary_per_glider.items():
                for j, g_b_id in self.summary_per_glider.items():
                    ref_df = self.summary_per_glider[g_id]
                    comp_df = self.summary_per_glider[g_b_id]

                    paired_df = find_closest_prof(ref_df, comp_df)
                    combined_summaries.append(paired_df)
            combined_summaries = pd.concat(combined_summaries, ignore_index=True)

        return

    def align_to_target(self, target="None"):
        """Align all datasets to a target dataset based on the summaries provided.

        Parameters
        ----------
        target : str
            The name of the target pipeline to align others to.
        """
        if not self._summary_ran:
            raise RuntimeError(
                "PipelineManager.summarise_all_profiles() must be run before alignment."
            )

        if target not in self.pipelines:
            raise ValueError(f"Target pipeline '{target}' not found.")

        target_data = self._contexts[target]["data"]

        # add summary data to data

        # rename variables in target_data based on alias in alignment_map
        # Apply alias renaming for alignment vars
        renamed_vars = {
            alias: std
            for std, alias_map in self.alignment_map.items()
            if (alias := alias_map.get(target)) and alias in target_data
        }
        print(f"[Pipeline Manager] Renaming variables: {renamed_vars}")
        target_data = target_data.rename(renamed_vars)
        target_summary = self.summary_per_glider[target].reset_index()

        r2_results_all = []

        # Determine which variables to align
        alignment_vars = list(self.alignment_map.keys())

        # get all pipeline names and data from self._contexts
        data = {
            name: (ctx["data"], self.summary_per_glider[name])
            for name, ctx in self._contexts.items()
        }

        for ancillary_name, (ancillary_data, ancillary_summary) in data.items():
            if ancillary_name == target:
                continue

            print(
                f"[Pipeline Manager] Aligning '{ancillary_name}' to target '{target}'..."
            )

            # Step 1: Find matched profile pairs
            paired_df = find_closest_prof(ancillary_summary, target_summary)
            paired_df = paired_df.rename(
                columns={
                    "glider_a_PROFILE_NUMBER": f"{ancillary_name}_PROFILE_NUMBER",
                    "glider_b_PROFILE_NUMBER": f"{target}_PROFILE_NUMBER",
                    "glider_b_time_diff": f"time_diff_hr_{ancillary_name}",
                    "glider_b_distance_km": f"distance_km_{ancillary_name}",
                }
            )
            print(f"[Pipeline Manager] Found {len(paired_df)} matched profiles.")

            if paired_df.empty:
                continue

            # Step 2: Interpolate and bin all datasets involved
            binned_dfs = {}
            for glider_name, gilder_data in [
                (target, target_data),
                (ancillary_name, ancillary_data),
            ]:
                print(f"[Pipeline Manager] Binning data for {glider_name}...")
                df_interp = interpolate_DEPTH(gilder_data)
                df_binned = aggregate_vars(df_interp, alignment_vars)
                binned_dfs[glider_name] = df_binned

            # Step 3: Merge target + source binned values at depth
            merged_df = merge_depth_binned_profiles(
                target_name=target, binned_dfs=binned_dfs
            )

            # Step 4: Filter merged_df for only profile pairs in matched list
            merged_df = merged_df[
                merged_df[f"{target}_PROFILE_NUMBER"].isin(
                    paired_df[f"{target}_PROFILE_NUMBER"]
                )
                & merged_df[f"{ancillary_name}_PROFILE_NUMBER"].isin(
                    paired_df[f"{ancillary_name}_PROFILE_NUMBER"]
                )
            ]

            # Step 5: Compute R² per profile-pair for all alignment variables
            r2_df = compute_r2_for_merged_profiles(
                merged_df=merged_df,
                variables=alignment_vars,
                target_name=target,
                other_names=[ancillary_name],
            )

            r2_results_all.append(r2_df)

        # Combine and store
        final_r2 = (
            pd.concat(r2_results_all, ignore_index=True)
            if r2_results_all
            else pd.DataFrame()
        )
        self.r2_alignment_results = final_r2
