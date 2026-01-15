"""Pipeline Class"""

import yaml
import pandas as pd
import numpy as np
import xarray as xr
import os
import logging
import datetime as _dt
from graphviz import Digraph

from toolbox.utils.config_mirror import ConfigMirrorMixin

from toolbox.steps import (
    create_step,
    STEP_CLASSES,
    STEP_DEPENDENCIES
)

_PIPELINE_LOGGER_NAME = "toolbox.pipeline"

def _setup_logging(log_file=None, level=logging.INFO):
    """Set up logging for the entire pipeline."""
    logger = logging.getLogger(_PIPELINE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler if specified
    if log_file:
        log_file = os.path.abspath(log_file)        # absolute path
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_file)

    return logger

class Pipeline(ConfigMirrorMixin):
    """
    Config-aware pipeline that can:
      - Load config YAML into private self._parameters
      - Keep global_parameters mirrored to _parameters['pipeline']
      - Build, run, and export steps as before
    """

    def __init__(self, config_path=None):
        """Initialize pipeline with optional config file"""
        self.steps = []  # hierarchical step configs
        self.graph = Digraph("Pipeline", format="png", graph_attr={"rankdir": "TB"})
        self.global_parameters = {}  # mirrors _parameters["pipeline"]
        self._context = None

        # initialise config mirror system
        self._init_config_mirror()

        if config_path:
            self.load_config_from_file(config_path, mirror_keys=["pipeline"])
            # set convenience alias for user-facing access
            self.global_parameters = self._parameters.get("pipeline", {})
            # build steps from loaded config
            self.logger = _setup_logging(self.global_parameters.get("log_file"))
            self.build_steps(self._parameters.get("steps", []))
            self.logger.info("Pipeline initialised")

    def build_steps(self, steps_config, parent_name=None):
        """Recursively build steps from configuration"""
        for step in steps_config:
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
            raise ValueError(
                f"Step '{step_name}' is not recognized or missing @register_step."
            )

        step_config = {
            "name": step_name,
            "parameters": parameters or {},
            "diagnostics": diagnostics,
            "substeps": [],
        }

        if parent_name:
            parent = self._find_step(self.steps, parent_name)
            if parent:
                parent["substeps"].append(step_config)
            else:
                raise ValueError(f"Parent step '{parent_name}' not found.")
        else:
            self.steps.append(step_config)

        self.logger.info(f"Step '{step_name}' added successfully!")

        if run_immediately:
            self.logger.info(f"Running step '{step_name}' immediately.")
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
        self.logger.info(f"Executing: {step.name}")
        return step.run()

    def run_last_step(self):
        """Runs only the most recently added step"""
        if not self.steps:
            self.logger.info("No steps to run.")
            return
        last_step = self.steps[-1]
        self.logger.info(f"Running last step: {last_step['name']}")
        self._context = self.execute_step(last_step, self._context)

    def run(self):
        """Runs the entire pipeline"""
        for step in self.steps:
            self._context = self.execute_step(step, self._context)

        if self.global_parameters.get("visualisation", False):
            self.visualise_pipeline()

    def visualise_pipeline(self):
        """Generates a visualisation of the pipeline execution"""
        self.graph.clear()

        def add_to_graph(step_config, parent_name=None, step_order=None):
            step_name = step_config["name"]
            diagnostics = step_config.get("diagnostics", False)
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
            if step_order and len(step_order) > 1:
                for i in range(len(step_order) - 1):
                    self.graph.edge(step_order[i], step_order[i + 1])
            substep_order = []
            for substep in step_config.get("substeps", []):
                substep_order.append(substep["name"])
                add_to_graph(substep, parent_name=step_name, step_order=substep_order)

        for step in self.steps:
            step_order = [step["name"]]
            add_to_graph(step, step_order=step_order)
        self.graph.render("pipeline_visualisation", view=True)

    def generate_config(self):
        """Generate a configuration dictionary from the current pipeline setup"""
        cfg = {
            "pipeline": self.global_parameters,
            "steps": self.steps,
        }
        # Keep private config in sync
        self._parameters.update(cfg)
        return cfg

    def export_config(self, output_path="generated_pipeline.yaml"):
        """Write current config to file (respects private _parameters)"""
        cfg = self.generate_config()
        with open(output_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.logger.info(f"Pipeline config exported → {output_path}")
        return cfg

    def save_config(self, path="pipeline_config.yaml"):
        """Save the canonical private config (same as manager.save_config)."""
        # ensure _parameters is up to date
        self._parameters.update(self.generate_config())
        super().save_config(path)