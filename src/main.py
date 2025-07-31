"""Entry point for the toolbox package."""

from toolbox.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline("config.yaml")
    pipeline.run()
