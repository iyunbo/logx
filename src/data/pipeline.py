"""create the pipeline
"""

from kedro.pipeline import Pipeline, node

from .nodes import parse_log, make_dataloader


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                parse_log,
                ["params:input_dir",
                 "params:log_format",
                 "params:log_file",
                 "params:parsing_result_dir"],
                "structured_csv",
            ),
            node(
                make_dataloader,
                "structured_csv",
                ["dataloader", "num_classes"],
            )
        ]
    )
