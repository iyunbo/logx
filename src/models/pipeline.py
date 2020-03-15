"""create the pipeline
"""

from kedro.pipeline import Pipeline, node

from .nodes import train, predict


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train,
                ["dataloader", "num_classes"],
                "model_file",
            ),
            node(
                predict,
                ["num_classes", "model_file"],
                None,
            )
        ]
    )
