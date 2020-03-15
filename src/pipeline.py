"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from .data import pipeline as de
from .models import pipeline as ds


###########################################################################
# Here you can find an example pipeline, made of two modular pipelines.
#
# Delete this when you start working on your own Kedro project as
# well as pipelines/data_science AND pipelines/data_engineering
# -------------------------------------------------------------------------


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }
