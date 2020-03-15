"""create the pipeline
"""

from kedro.pipeline import Pipeline, node

from .nodes import parse_log, make_dataloader


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                parse_log,
                None,
                "structured_csv",
            ),
            node(
                make_dataloader,
                "structured_csv",
                ["dataloader", "num_classes"],
            )
        ]
    )
