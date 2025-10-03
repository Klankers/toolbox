# Developer Guide

This page explains how to add a new **step** to the Toolbox (e.g., a new
processing stage, validation routine, or export).

## What is a “step”?
A step is a stage in the pipeline that can be defined via Python, and configured via the Pipelines Config file. Examples of steps include:
- I/O Steps (e.g., reading from a [load_data.py](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/steps/custom/load_data/index.html), containing `Load OG1`, or `export.py`, containing `Data Export`)
- Variable Processing Steps (e.g., [salinity.py](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/steps/custom/variables/salinity/index.html), containing `QC: Salinity` and `ADJ: Salinity`)
- Data Processing Steps (e.g., [derive_ctd.py](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/steps/custom/derive_ctd/index.html), containing `Find Profiles`)

Steps are not limited to one per file - in fact, a single file can contain multiple steps. For example, the [salinity.py](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/steps/custom/variables/salinity/index.html) file contains both a QC and an ADJ step.

## How to add a new step
1. Create a new Python file in the appropriate directory under `src/toolbox/steps/custom/`.
   **NOTE**: if you are creating a step for specific vairables - such as a salinity QC step - then it should go in the `variables` subdirectory.
2. Define a new class for your step, inheriting from `BaseStep` (or another appropriate base class, such as `VariableStep` or `DataStep`).
   ```python
    from toolbox.steps.base import BaseStep
    class MyNewStep(BaseStep):
        ...
    ```
3. Define the step_name attribute, which is the name that will be used in the Pipelines Config file to refer to this step.
   ```python
    class MyNewStep(BaseStep):
        step_name = "My New Step"
        ...
    ```
4. Register the step using the `@register_step` decorator.
   ```python
    from toolbox.steps import register_step
    @register_step
    class MyNewStep(BaseStep):
        step_name = "My New Step"
        ...
    ```
    This ensures that the step is discoverable by the Pipeline Manager, as well as allowing you do define other classes in the same file without registering them.
5. Implement the `run` method, which contains the logic for your step. This method should take no arguments other than `self`, and should return a `self.context` object.
   ```python
    class MyNewStep(BaseStep):
        step_name = "My New Step"
        
        def run(self):
            # Your processing logic here
            return self.context        
    ```
6. Optionally, implement the `generate_diagnostics` method if your step produces any diagnostic plots or outputs.
   ```python
    class MyNewStep(BaseStep):
        step_name = "My New Step"
        
        def run(self):
            # Your processing logic here
            return self.context
        
        def generate_diagnostics(self):
            # Your diagnostics logic here
            pass
    ```
    There are already default methods for generating common diagnostics, such as time series plots and scatter plots. See the [utils.diagnostics documentation](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/utils/diagnostics/index.html) for more information.

7. Add the step to your Pipelines Config file, using the `step_name` you defined in step 3.
   ```yaml
    # Pipeline Configuration
    pipeline:
    name: "My Pipeline"
    description: "A pipeline for demonstration purposes"
    # Steps in the pipeline
    steps:
    - name: "My New Step"
      parameters:
        param1: value1
        param2: value2
   ```

8. Any parameters defined in the `parameters` section of the config file will be passed to your step as attributes. You can access them in your `run` method using `self.param1`, `self.param2`, etc.
   **NOTE** This is handled automatically by the `BaseStep` class. More information can be found in the [BaseStep documentation](https://noc-obg-autonomy.github.io/toolbox/api/toolbox/steps/base_step/index.html).  