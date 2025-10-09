"""Pipeline Class"""

import yaml
import pandas as pd
import numpy as np
import xarray as xr
import os
import datetime as _dt

from toolbox.utils.diagnostics import (
    summarising_profiles,
    plot_distance_time_grid,
    plot_glider_pair_heatmap_grid,
)
from toolbox.utils.alignment import (
    interpolate_DEPTH,
    aggregate_vars,
    merge_pairs_from_filtered_aggregates,
    filter_xarray_by_profile_ids,
    find_profile_pair_metadata,
    compute_r2_for_merged_profiles_xr,
    plot_r2_heatmaps_per_pair,
    plot_pair_scatter_grid,
    collect_xy_from_r2_ds,
    fit_linear_map,
    _inject_aligned_into_target_grid,
)

from toolbox.utils.export import (
    _append_history,
    _enrich_combined_attrs,
    _enrich_processed_attrs,
    _copy_coord_attrs,
    _copy_global_attrs,
)

from toolbox.utils.validation import validate

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
            print(f"[Pipeline Manager] Heatmap grid plotted in {end_time - start_time}")

        return

    def preview_alignment(self, target="None"):
        """
        Align all datasets to a target dataset and compute R² against ancillary sources.

        This version:
        - Renames each pipeline's variables to the standard names (from alignment_map)
        - Runs interpolate + aggregate ONCE per pipeline and caches the results
        - Uses the cached medians for pairing/merging/R²
        - Populates exportable handles for raw/processed/lite data
        """

        # === PRECONDITIONS ===
        if not self._summary_ran:
            raise RuntimeError("Run summarise_all_profiles() before alignment.")

        if target not in self.pipelines:
            raise ValueError(f"Target pipeline '{target}' not found.")

        # === CONFIG ===
        alignment_vars = list(self.alignment_map.keys())
        self.r2_datasets = {}  # Reset R² result container

        # ---- Helper: alias -> std renamer for a given pipeline name ----
        def _rename_to_standard(name: str, ds):
            rename_map = {
                alias: std
                for std, alias_map in self.alignment_map.items()
                if (alias := alias_map.get(name)) and alias in ds
            }
            return ds.rename(rename_map) if rename_map else ds, rename_map

        if not hasattr(self, "processed_per_glider"):
            self.processed_per_glider = {}

        # export registry the rest of your workflow can use later to write files
        if not hasattr(self, "_exportables"):
            self._exportables = {"raw": {}, "processed": {}, "lite": {}}

        # === COLLECT: standardised & processed datasets for ALL pipelines (target + ancillaries) ===
        for name, ctx in self._contexts.items():
            raw_ds = ctx["data"]

            # Keep a pointer to raw data for export (no copy)
            self._exportables["raw"][name] = raw_ds

            # If we already processed this pipeline, skip recomputation
            if (
                name in self.processed_per_glider
                and "agg" in self.processed_per_glider[name]
            ):
                continue

            # 1) rename variables to standard names
            ds_std, used_map = _rename_to_standard(name, raw_ds)

            # 2) interpolate depth
            print(f"[Pipeline Manager] Interpolating DEPTH for '{name}'...")
            ds_interp = interpolate_DEPTH(ds_std)

            # 3) aggregate medians (2-D by PROFILE_NUMBER × DEPTH_bin)
            print(f"[Pipeline Manager] Aggregating medians for '{name}'...")
            ds_agg = aggregate_vars(ds_interp, alignment_vars)

            # store in cache
            self.processed_per_glider[name] = {
                "renamed": ds_std,
                "interp": ds_interp,
                "agg": ds_agg,
            }

            # make processed export handle available
            self._exportables["processed"][name] = ds_agg
            if "lite" not in self._exportables:
                self._exportables["lite"] = {}

        # Prepare target objects
        target_summary = self.summary_per_glider[target].reset_index()
        target_agg = self.processed_per_glider[target]["agg"]

        # === LOOP: align each ancillary to target using the cached medians ===
        for ancillary_name, ctx in self._contexts.items():
            if ancillary_name == target:
                continue

            print(
                f"\n[Pipeline Manager] Aligning '{ancillary_name}' to target '{target}'..."
            )

            ancillary_summary = self.summary_per_glider[ancillary_name]
            if ancillary_summary.index.names[0] is not None:
                ancillary_summary = ancillary_summary.reset_index()

            # === STEP 1: Find Matched Profile Pairs ===
            paired_df = find_profile_pair_metadata(
                df_target=target_summary,
                df_ancillary=ancillary_summary,
                target_name=target,
                ancillary_name=ancillary_name,
                time_thresh_hr=self.settings.get("diagnostics", {})
                .get("matchup_thresholds", {})
                .get("max_time_threshold", 12),
                dist_thresh_km=self.settings.get("diagnostics", {})
                .get("matchup_thresholds", {})
                .get("max_distance_threshold", 20),
            )

            if paired_df.empty:
                print(
                    f"[Pipeline Manager] No matched profiles between {target} and {ancillary_name}."
                )
                continue

            print(f"[Pipeline Manager] Found {len(paired_df)} matched profile pairs.")

            # === STEP 2: Use CACHED aggregated medians ===
            binned_ds = {
                target: target_agg,
                ancillary_name: self.processed_per_glider[ancillary_name]["agg"],
            }

            # === STEP 3: Filter Datasets by Matched Profile IDs ===
            filtered_ds = {}
            for glider_name, agg_ds in [
                (target, binned_ds[target]),
                (ancillary_name, binned_ds[ancillary_name]),
            ]:
                profile_ids = paired_df[f"{glider_name}_PROFILE_NUMBER"].values
                filtered_ds[glider_name] = filter_xarray_by_profile_ids(
                    ds=agg_ds,
                    profile_id_var="PROFILE_NUMBER",
                    valid_ids=profile_ids,
                )

            # === STEP 4: Build pairwise merged dataset ===
            merged = merge_pairs_from_filtered_aggregates(
                paired_df=paired_df,
                agg_target=filtered_ds[target],
                agg_anc=filtered_ds[ancillary_name],
                target_name=target,
                ancillary_name=ancillary_name,
                variables=alignment_vars,  # the raw names; helper will use median_{var}
            )

            print("[Align] Merged dims:", merged.dims)
            print("[Align] Vars:", list(merged.data_vars))

            # === STEP 5: Compute R² ===
            print(f"[Pipeline Manager] Computing R² for '{ancillary_name}'...")
            r2_ds = compute_r2_for_merged_profiles_xr(
                ds=merged,
                variables=alignment_vars,
                target_name=target,
                ancillary_name=ancillary_name,
            )

            self.r2_datasets[ancillary_name] = r2_ds

            # add to cache
            self._exportables["processed"][f"{target}_vs_{ancillary_name}"] = r2_ds
            print(
                f"[Pipeline Manager] R² dataset stored for '{target}' vs '{ancillary_name}'."
            )

        print("\n[Pipeline Manager] Alignment complete for all datasets.")

        # Set R² thresholds
        r2_thresholds = self.settings.get("alignment", {}).get(
            "r2_thresholds", [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        )

        # Call the plotting function
        # r2_datasets produced by align_to_target
        plot_r2_heatmaps_per_pair(
            r2_datasets=self.r2_datasets,
            variables=list(self.alignment_map.keys()),
            target_name=target,  # e.g. "Doombar"
            r2_thresholds=r2_thresholds,
            time_thresh_hr=self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_time_threshold", 12),
            dist_thresh_km=self.settings.get("diagnostics", {})
            .get("matchup_thresholds", {})
            .get("max_distance_threshold", 20),
            figsize=(9, 6),
            save_plots=self.settings.get("alignment", {}).get("save_plots", False),
            output_path=self.settings.get("alignment", {}).get(
                "plot_output_path", "r2_heatmap_grid.png"
            ),
            show_plots=self.settings.get("alignment", {}).get("show_plots", True),
        )

    def fit_and_save_to_target(
        self,
        target,
        out_dir=None,
        variable_r2_criteria=None,
        max_time_hr=None,
        max_dist_km=None,
        ancillaries=None,
        overwrite=False,
        show_plots=True,
    ):
        """
        Fit ancillary variables to target datasets using profile-pair medians and per-variable R² criteria.
        """
        if out_dir is None:
            # get directory from settings or create timestamped dir
            datetime_str = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_dir = self.settings.get("alignment", {}).get(
                "output_path", datetime_str
            )
        if not getattr(self, "r2_datasets", None):
            raise RuntimeError("Run preview_alignment() before fitting.")
        if target not in self.pipelines or target not in self._contexts:
            raise ValueError(f"Target '{target}' not available in manager contexts.")

        alignment_vars = list(self.alignment_map.keys())
        if variable_r2_criteria is None:
            variable_r2_criteria = self.settings.get("alignment", {}).get(
                "variable_r2_criteria", {}
            )
        missing = [v for v in alignment_vars if v not in variable_r2_criteria]
        if missing:
            raise ValueError(f"Missing R² thresholds for variables: {missing}")

        os.makedirs(out_dir, exist_ok=True)

        all_sources = [n for n in self.pipelines.keys() if n != target]
        anc_list = list(ancillaries) if ancillaries else all_sources

        if show_plots and hasattr(self, "plot_pair_scatter_grid"):
            try:
                plot_pair_scatter_grid(
                    r2_datasets=self.r2_datasets,
                    variables=alignment_vars,
                    target_name=target,
                    variable_r2_criteria=variable_r2_criteria,
                    max_time_hr=max_time_hr,
                    max_dist_km=max_dist_km,
                )
            except Exception as e:
                print(f"[Fit] (Plot) Skipped grid due to: {e}")

        def _alias_map_for(source_name):
            return {
                alias: std
                for std, mapping in self.alignment_map.items()
                if (alias := mapping.get(source_name))
            }

        saved_paths = {}
        fits_summary = {}

        for anc in anc_list:
            if anc not in self._contexts:
                print(f"[Fit] Skipping '{anc}' (no context).")
                continue

            print(f"\n[Fit] === {anc} → align to {target} ===")
            r2_ds = self.r2_datasets.get(anc)
            if r2_ds is None or not isinstance(r2_ds, xr.Dataset):
                print(f"[Fit] No R² dataset for '{anc}'. Skipping.")
                continue

            anc_ds = self._contexts[anc]["data"]
            rename_map = _alias_map_for(anc)
            anc_ds_std = anc_ds.rename(rename_map) if rename_map else anc_ds

            # Compute per-variable fits
            anc_fits = {}
            for var in alignment_vars:
                x, y = collect_xy_from_r2_ds(
                    r2_ds,
                    var=var,
                    target_name=target,
                    ancillary_name=anc,
                    r2_min=variable_r2_criteria.get(var),
                    time_max=max_time_hr,
                    dist_max=max_dist_km,
                )
                fit = fit_linear_map(x, y)
                anc_fits[var] = fit
                print(
                    f"[Fit] {anc}:{var} slope={fit['slope']:.4g} intercept={fit['intercept']:.4g} "
                    f"R²={fit['r2']:.3f} N={fit['n']}"
                )

            # Apply to full ancillary dataset (creates {VAR}_ALIGNED_TO_{target})
            ds_out = anc_ds_std.copy()
            created_vars = []
            for var in alignment_vars:
                if var not in ds_out:
                    print(f"[Fit] [{anc}] missing '{var}' — skip.")
                    continue
                slope = anc_fits[var]["slope"]
                intercept = anc_fits[var]["intercept"]
                npts = anc_fits[var]["n"]
                out_name = f"{var}_ALIGNED_TO_{target}"

                aligned = slope * ds_out[var] + intercept
                aligned = aligned.astype(ds_out[var].dtype, copy=False)
                aligned.name = out_name
                aligned.attrs.update(
                    {
                        "long_name": f"{var} aligned to {target}",
                        "alignment_target": target,
                        "alignment_source": anc,
                        "alignment_slope": float(slope),
                        "alignment_intercept": float(intercept),
                        "alignment_fit_points": int(npts),
                        "alignment_generated": _dt.datetime.utcnow().isoformat() + "Z",
                    }
                )
                ds_out[out_name] = aligned
                created_vars.append(out_name)

            if not created_vars:
                print(f"[Fit] No aligned variables for '{anc}'. Skipping save.")
                continue

            # Keep aligned vars in memory
            try:
                self._contexts[anc]["data"] = ds_out
            except Exception:
                pass

            # >>> NEW: immediately build & cache aggregated aligned medians for this ancillary
            try:
                _ = self._aggregate_aligned_vars_for_ancillary(
                    anc_name=anc, target=target, vars_to_aggregate=alignment_vars
                )  # populates self.processed_per_glider[anc][f'agg_aligned_to_{target}']
            except Exception as e:
                print(
                    f"[Fit] Warning: failed to cache aggregated aligned medians for '{anc}': {e}"
                )

            # Save per-ancillary file
            out_path = os.path.join(out_dir, f"{anc}_aligned_to_{target}.nc")
            if (not overwrite) and os.path.exists(out_path):
                print(f"[Fit] File exists, not overwriting: {out_path}")
            else:
                encoding = {
                    name: {"zlib": True, "complevel": 2} for name in created_vars
                }
                try:
                    ds_out.to_netcdf(out_path, encoding=encoding)
                    print(f"[Fit] Saved: {out_path}")
                    saved_paths[anc] = out_path
                    fits_summary[anc] = anc_fits
                except Exception as e:
                    print(f"[Fit] Failed to save '{anc}': {e}")

            # Cache fits for metadata
            if anc in self.processed_per_glider:
                self.processed_per_glider[anc][
                    f"last_fit_to_target_{target}"
                ] = anc_fits

        return {"paths": saved_paths, "fits": fits_summary}

    def validate_with_device(self, target="None", **overrides):
        """
        Run the validation workflow using settings['validation'].
        Optionally pass keyword overrides (e.g., show_plots=False) for this call only.

        Examples:
            mngr.validate_with_device("Doombar")
            mngr.validate_with_device("Doombar", show_plots=False, apply_and_save=True)
        """
        # Fast path: no overrides → just call through
        if not overrides:
            validate(self, target=target)
            return

        # One-shot overrides: temporarily update settings['validation']
        vcfg_orig = dict(self.settings.get("validation", {}))  # shallow copy
        try:
            vcfg = self.settings.setdefault("validation", {})
            vcfg.update(overrides)
            validate(self, target=target)
        finally:
            # restore original validation config
            self.settings["validation"] = vcfg_orig

    def fit_to_device(self, target="None"):
        """
        Fit TARGET variables to a validation device using profile-pair medians and per-variable R² criteria.
        The mapping is fit as: device = slope * target + intercept, then applied to the FULL target dataset
        to create new variables `{VAR}_ALIGNED_TO_{DEVICE}`.

        Reads options from self.settings['validation']:
        validation:
            device_name: "<device label>"
            variable_names: ["CNDC","TEMP", ...]        # optional; defaults to alignment_map keys
            variable_r2_criteria: {CNDC: 0.95, TEMP: 0.9, ...}
            max_time_threshold: <float>
            max_distance_threshold: <float>
            save_plots: <bool>
            show_plots: <bool>
            plot_output_path: "<file or dir>"
            apply_and_save: <bool>
            output_path: "<dir or empty for timestamped dir>"

        Returns
        -------
        dict with:
        - "path": output NetCDF (if saved)
        - "fits": {var: {slope, intercept, r2, n}, ...}
        - "device_name": device label used
        """
        # --- Preconditions ---
        if target not in self.pipelines:
            raise ValueError(f"Target pipeline '{target}' not found.")
        if target not in self._contexts:
            raise ValueError(f"Target pipeline '{target}' has no context data.")

        # --- Validation config ---
        vcfg = self.settings.get("validation", {}) or {}
        device_name = vcfg.get("device_name", "DEVICE")
        variables = vcfg.get("variable_names", list(self.alignment_map.keys()))
        var_r2_criteria = vcfg.get("variable_r2_criteria", {}) or {}
        max_time_hr = vcfg.get("max_time_threshold", None)
        max_dist_km = vcfg.get("max_distance_threshold", None)
        show_plots = bool(vcfg.get("show_plots", True))
        save_plots = bool(vcfg.get("save_plots", False))
        plot_output_path = vcfg.get("plot_output_path", "device_fit_scatter_grid.png")
        apply_and_save = bool(vcfg.get("apply_and_save", False))
        out_dir = vcfg.get("output_path", "") or ""

        # Validate thresholds exist for all requested variables
        missing = [v for v in variables if v not in var_r2_criteria]
        if missing:
            raise ValueError(
                f"[Fit→Device] R² threshold missing for variables: {missing}"
            )
        print(f"[Fit→Device] Using device='{device_name}', variables={variables}")
        print(f"[Fit→Device] R² thresholds: {var_r2_criteria}")

        # --- Ensure we have the R² dataset for TARGET vs DEVICE ---
        # This will run the whole validation pipeline (load device, pair, aggregate, merge, compute R²)
        from .utils.validation import validate  # adjust import if your layout differs

        val_res = validate(self, target=target)
        r2_ds = val_res.get("r2_ds", None)
        if r2_ds is None or not isinstance(r2_ds, xr.Dataset):
            raise RuntimeError(
                "[Fit→Device] No R² dataset available from validation()."
            )

        # --- QA scatter grid (X=device, Y=target) before fitting ---
        if show_plots or save_plots:
            try:
                # plot_pair_scatter_grid expects a dict of {ancillary_name: ds}
                ds_map = {device_name: r2_ds}
                fig, _ = plot_pair_scatter_grid(
                    r2_datasets=ds_map,
                    variables=variables,
                    target_name=target,
                    variable_r2_criteria=var_r2_criteria,
                    max_time_hr=max_time_hr,
                    max_dist_km=max_dist_km,
                    ancillaries_order=[device_name],
                )
                if save_plots:
                    # If path looks like a directory, drop a default filename into it
                    out_is_dir = (plot_output_path.endswith(os.sep)) or (
                        os.path.isdir(plot_output_path)
                    )
                    if out_is_dir:
                        os.makedirs(plot_output_path, exist_ok=True)
                        fout = os.path.join(
                            plot_output_path, f"{target}_vs_{device_name}_fit_grid.png"
                        )
                    else:
                        os.makedirs(
                            os.path.dirname(plot_output_path) or ".", exist_ok=True
                        )
                        fout = plot_output_path
                    fig.savefig(fout, dpi=300)
                    print(f"[Fit→Device] Saved scatter grid to: {fout}")
                if not show_plots:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
            except Exception as e:
                print(f"[Fit→Device] (Plot) Skipped grid due to: {e}")

        # --- Compute fits to map TARGET → DEVICE for each variable ---
        # collect_xy_from_r2_ds returns (X=device, Y=target). For TARGET→DEVICE we invert to (x=target, y=device).
        fits = {}
        for var in variables:
            X_dev, Y_tgt = collect_xy_from_r2_ds(
                r2_ds,
                var=var,
                target_name=target,
                ancillary_name=device_name,
                r2_min=var_r2_criteria.get(var),
                time_max=max_time_hr,
                dist_max=max_dist_km,
            )
            # invert orientation for target->device mapping
            x = Y_tgt  # target
            y = X_dev  # device
            info = fit_linear_map(x, y)  # fits y_device = a * x_target + b
            fits[var] = info
            print(
                f"[Fit→Device] {var}: device ≈ {info['slope']:.4g}·target + {info['intercept']:.4g} "
                f"(R²={info['r2']:.3f}, N={info['n']})"
            )

        # --- Apply mapping to the FULL target dataset (create {var}_ALIGNED_TO_{device}) ---
        # Rename target variables to standard names based on alignment_map aliases
        target_ds_raw = self._contexts[target]["data"]
        rename_map = {
            alias: std
            for std, alias_map in self.alignment_map.items()
            if (alias := alias_map.get(target)) and alias in target_ds_raw
        }
        target_ds_std = (
            target_ds_raw.rename(rename_map) if rename_map else target_ds_raw
        )

        ds_out = target_ds_std.copy()
        created = []
        for var, info in fits.items():
            if var not in ds_out:
                print(f"[Fit→Device] Target missing variable '{var}' — skipping.")
                continue
            slope, intercept, npts = info["slope"], info["intercept"], info["n"]
            out_name = f"{var}_ALIGNED_TO_{device_name}"
            aligned = (slope * ds_out[var] + intercept).astype(
                ds_out[var].dtype, copy=False
            )
            aligned.name = out_name
            aligned.attrs.update(
                {
                    "long_name": f"{var} aligned to {device_name}",
                    "alignment_target": target,
                    "alignment_reference_device": device_name,
                    "alignment_direction": "target_to_device",
                    "alignment_slope": float(slope),
                    "alignment_intercept": float(intercept),
                    "alignment_fit_points": int(npts),
                }
            )
            ds_out[out_name] = aligned
            created.append(out_name)

        if not created:
            print("[Fit→Device] No aligned variables were created — nothing to save.")
            return {"path": None, "fits": fits, "device_name": device_name}

        # update processed_per_glider for potential later use
        if target not in self.processed_per_glider:
            self.processed_per_glider[target] = {}
        self.processed_per_glider[target][f"last_fit_to_device_{device_name}"] = fits

        return {"fits": fits, "device_name": device_name}

    def _aggregate_aligned_vars_for_ancillary(
        self, anc_name: str, target: str, vars_to_aggregate: list[str]
    ):
        """
        Return an aggregated dataset of the ancillary's *already-aligned* variables,
        with medians by (PROFILE_NUMBER, DEPTH_bin) for variables named:
            {VAR}_ALIGNED_TO_{target}

        Cache key:
            self.processed_per_glider[anc_name]['agg_aligned_to_{target}']

        If the cache is missing, compute it from the ancillary context dataset
        (which contains the {VAR}_ALIGNED_TO_{target} variables written by fit_and_save_to_target()).
        """
        if anc_name not in self._contexts:
            raise ValueError(f"Unknown ancillary '{anc_name}'.")

        cache_key = f"agg_aligned_to_{target}"
        # 1) Use cache if we already built it
        if (
            anc_name in self.processed_per_glider
            and cache_key in self.processed_per_glider[anc_name]
        ):
            return self.processed_per_glider[anc_name][cache_key]

        # 2) Otherwise, compute once from the ancillary context dataset
        ds_anc = self._contexts[anc_name]["data"]
        aligned_base_vars = []
        for var in vars_to_aggregate:
            name = f"{var}_ALIGNED_TO_{target}"
            if name in ds_anc:
                aligned_base_vars.append(name)
            else:
                print(f"[AggregateAligned] {anc_name}: missing '{name}', skipping.")

        if not aligned_base_vars:
            raise RuntimeError(
                f"[AggregateAligned] No aligned variables found in '{anc_name}' for target '{target}'."
            )

        # Ensure required coords exist, then aggregate ONLY the aligned vars
        ds_work = ds_anc
        if (
            ("DEPTH_bin" not in ds_work)
            or ("PROFILE_NUMBER" not in ds_work)
            or ("N_MEASUREMENTS" not in ds_work.dims)
        ):
            ds_work = interpolate_DEPTH(ds_work)

        agg_aligned = aggregate_vars(
            ds_work,
            aligned_base_vars,  # e.g. ["TEMP_ALIGNED_TO_Doombar", ...]
            profile_dim="PROFILE_NUMBER",
            bin_dim="DEPTH_bin",
        )
        # Note: aggregate_vars will produce vars like "median_TEMP_ALIGNED_TO_Doombar"

        # Cache for future calls
        self.processed_per_glider.setdefault(anc_name, {})
        self.processed_per_glider[anc_name][cache_key] = agg_aligned
        return agg_aligned

    def _build_aligned_vars_on_target_grid(
        self, target: str
    ) -> dict[str, xr.DataArray]:
        """
        Using self.r2_datasets and each ancillary's *already aligned* variables
        ({VAR}_ALIGNED_TO_{target}), create DataArrays on the target's
        (PROFILE_NUMBER, DEPTH_bin) grid:

            median_{VAR}_{ANCILLARY}_ALIGNED_TO_{target}

        Returns
        -------
        dict[str, xr.DataArray]
        """
        if (
            target not in self.processed_per_glider
            or "agg" not in self.processed_per_glider[target]
        ):
            raise RuntimeError(
                "Target aggregated medians not cached. Run preview_alignment() first."
            )
        ds_target_agg = self.processed_per_glider[target]["agg"]
        alignment_vars = list(self.alignment_map.keys())

        out = {}
        if not hasattr(self, "r2_datasets") or not isinstance(self.r2_datasets, dict):
            return out

        for anc, r2_ds in self.r2_datasets.items():
            if anc == target or not isinstance(r2_ds, xr.Dataset):
                continue

            try:
                anc_aligned_agg = self._aggregate_aligned_vars_for_ancillary(
                    anc_name=anc, target=target, vars_to_aggregate=alignment_vars
                )
            except Exception as e:
                print(
                    f"[Export] {anc}: could not build aggregated aligned medians: {e}"
                )
                continue

            for var in alignment_vars:
                try:
                    da = _inject_aligned_into_target_grid(
                        target_agg=ds_target_agg,
                        r2_ds=r2_ds,
                        target_name=target,
                        ancillary_name=anc,
                        var=var,
                        ancillary_aligned_agg=anc_aligned_agg,
                        bin_dim="DEPTH_bin",
                        pair_dim="PAIR_INDEX",
                    )
                    out[da.name] = da
                except Exception as e:
                    print(f"[Export] Injection skipped ({anc}:{var}): {e}")

        return out

    def save_outputs(self, target: str):
        """
        Save:
        - Raw datasets (as they arrived from each pipeline)
        - Processed datasets per pipeline (aggregated medians)
        - Combined target dataset (target medians + ancillary aligned medians on target grid)
        - Lite version of the combined dataset

        Also:
        - The TARGET processed file includes median_{VAR}_{ANC}_ALIGNED_TO_{target}.
        - Fit parameters (slope/intercept/R²/N) are written into global attrs.
        """
        saving_cfg = self.settings.get("saving", {}) or {}
        save_raw = bool(saving_cfg.get("save_raw", True))
        save_processed = bool(saving_cfg.get("save_processed", True))
        save_combined = bool(saving_cfg.get("save_combined_target", True))
        save_lite = bool(saving_cfg.get("save_lite", True))

        out_dir = saving_cfg.get("output_dir", "") or os.path.join(
            os.getcwd(), f"exports_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        )
        os.makedirs(out_dir, exist_ok=True)

        alignment_vars = list(self.alignment_map.keys())

        # ----- 1) Raw
        if save_raw:
            for name, ctx in self._contexts.items():
                ds_raw = ctx["data"]
                ds_to_save = ds_raw.copy(deep=False)
                ds_to_save.attrs = _append_history(
                    ds_to_save.attrs, f"Raw dataset exported for pipeline '{name}'."
                )
                path = os.path.join(out_dir, f"{name}_raw.nc")
                try:
                    ds_to_save.to_netcdf(path)
                    print(f"[Save] Raw → {path}")
                except Exception as e:
                    print(f"[Save] Raw FAILED for {name}: {e}")

        # Sanity
        if not hasattr(
            self, "processed_per_glider"
        ) or "agg" not in self.processed_per_glider.get(target, {}):
            raise RuntimeError(
                "processed_per_glider not populated. Run preview_alignment() first."
            )

        # ----- 2) Processed (aggregated medians) per pipeline
        if save_processed:
            for name, parts in self.processed_per_glider.items():
                ds_agg = parts.get("agg", None)
                if ds_agg is None or not isinstance(ds_agg, xr.Dataset):
                    print(f"[Save] Skipping processed export for '{name}' (no 'agg').")
                    continue

                ds_proc = ds_agg.copy()

                # If this is the TARGET, also add aligned ancillary medians on the target grid
                if name == target:
                    aligned_grid_vars = self._build_aligned_vars_on_target_grid(target)
                    for vname, da in aligned_grid_vars.items():
                        ds_proc[vname] = da

                    # Add fit parameters to METADATA (attrs) if we have them
                    fits_attrs = {}
                    for anc in self._contexts.keys():
                        if anc == target:
                            continue
                        fit_bucket = self.processed_per_glider.get(anc, {}).get(
                            f"last_fit_to_target_{target}", {}
                        )
                        if fit_bucket:
                            for var, info in fit_bucket.items():
                                fits_attrs[f"{anc}::{var}"] = {
                                    "slope": float(info.get("slope", float("nan"))),
                                    "intercept": float(
                                        info.get("intercept", float("nan"))
                                    ),
                                    "r2": float(info.get("r2", float("nan"))),
                                    "n": int(info.get("n", 0)),
                                }
                    if fits_attrs:
                        import json as _json

                        ds_proc.attrs["alignment_fits_json"] = _json.dumps(fits_attrs)

                # attrs & write
                ds_proc = _enrich_processed_attrs(
                    self, name, ds_proc, alignment_vars, bin_dim="DEPTH_bin"
                )
                ds_proc.attrs = _append_history(
                    ds_proc.attrs, f"Processed medians exported for pipeline '{name}'."
                )

                path = os.path.join(out_dir, f"{name}_processed_agg.nc")
                try:
                    ds_proc.to_netcdf(path)
                    print(f"[Save] Processed → {path}")
                except Exception as e:
                    print(f"[Save] Processed FAILED for {name}: {e}")

        # ----- 3) Combined target dataset
        if save_combined:
            ds_target_agg = self.processed_per_glider[target]["agg"]
            ds_comb = ds_target_agg.copy()

            # Add aligned ancillary medians (on target grid) — no re-fitting
            aligned_grid_vars = self._build_aligned_vars_on_target_grid(target)
            for vname, da in aligned_grid_vars.items():
                ds_comb[vname] = da

            # Persist fits to METADATA (attrs) — not as extra data variables
            fits_attrs = {}
            for anc in self._contexts.keys():
                if anc == target:
                    continue
                fit_bucket = self.processed_per_glider.get(anc, {}).get(
                    f"last_fit_to_target_{target}", {}
                )
                if fit_bucket:
                    for var, info in fit_bucket.items():
                        fits_attrs[f"{anc}::{var}"] = {
                            "slope": float(info.get("slope", float("nan"))),
                            "intercept": float(info.get("intercept", float("nan"))),
                            "r2": float(info.get("r2", float("nan"))),
                            "n": int(info.get("n", 0)),
                        }
            if fits_attrs:
                import json as _json

                ds_comb.attrs["alignment_fits_json"] = _json.dumps(fits_attrs)

            # Enrich & write
            ds_comb = _enrich_combined_attrs(self, target, ds_comb, alignment_vars)
            ds_comb.attrs = _append_history(
                ds_comb.attrs,
                f"Combined dataset exported for target '{target}' with ancillary aligned medians on target grid.",
            )
            path = os.path.join(out_dir, f"{target}_combined_agg.nc")
            try:
                ds_comb.to_netcdf(path)
                print(f"[Save] Combined target → {path}")
            except Exception as e:
                print(f"[Save] Combined target FAILED: {e}")

            # ----- 4) Lite
            if save_lite:
                suffix = f"_ALIGNED_TO_{target}"
                keep_vars = [v for v in ds_comb.data_vars if v.startswith("median_")]
                keep_vars += [v for v in ds_comb.data_vars if v.endswith(suffix)]
                ds_lite = ds_comb[keep_vars].copy()

                # coords
                keep_coords = [
                    c
                    for c in [
                        "PROFILE_NUMBER",
                        "DEPTH_bin",
                        "LATITUDE",
                        "LONGITUDE",
                        "TIME",
                    ]
                    if (c in ds_comb.coords or c in ds_comb)
                ]
                for c in keep_coords:
                    if c in ds_comb.coords:
                        ds_lite = ds_lite.assign_coords({c: ds_comb.coords[c]})
                    elif c in ds_comb:
                        ds_lite = ds_lite.assign_coords({c: ds_comb[c]})

                _copy_global_attrs(ds_lite, ds_comb, history_line="Lite view exported.")
                _copy_coord_attrs(ds_lite, ds_comb, coord_names=keep_coords)

                path_lite = os.path.join(out_dir, f"{target}_combined_agg_lite.nc")
                try:
                    ds_lite.to_netcdf(path_lite)
                    print(f"[Save] Lite → {path_lite}")
                except Exception as e:
                    print(f"[Save] Lite FAILED: {e}")
