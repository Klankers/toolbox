import os
import importlib
import pathlib
from .base_step import BaseStep, REGISTERED_STEPS

STEP_CLASSES = {}

STEP_DEPENDENCIES = {
    "QC: Salinity": ["Load OG1"],
}


def discover_steps():
    """Dynamically discover and import step modules from the custom directory."""
    base_dir = pathlib.Path(__file__).parent.resolve()
    custom_dir = base_dir / "custom"
    print(f"[Discovery] Scanning for step modules in {custom_dir}")

    for py_file in custom_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Convert file path to module path
        relative_path = py_file.resolve().relative_to(base_dir)
        module_name = ".".join(("toolbox.steps",) + relative_path.with_suffix("").parts)

        try:
            print(f"[Discovery] Importing step module: {module_name}")
            importlib.import_module(module_name)
        except Exception as e:
            print(f"[Discovery] Failed to import {module_name}: {e}")

    # Populate step classes
    STEP_CLASSES.update(REGISTERED_STEPS)
    for step_name in STEP_CLASSES:
        print(f"[Discovery] Registered step: {step_name}")


# Auto-discover steps on import
discover_steps()


def create_step(step_config, _context):
    """Create a step instance based on the provided configuration."""
    step_name = step_config["name"]
    step_class = STEP_CLASSES.get(step_name)
    if not step_class:
        raise ValueError(
            f"Step '{step_name}' not recognized or missing @register_step."
        )

    return step_class(
        name=step_name,
        parameters=step_config.get("parameters", {}),
        diagnostics=step_config.get("diagnostics", False),
        context=_context,
    )
