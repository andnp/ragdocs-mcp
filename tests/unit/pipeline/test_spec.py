from searchkernel.pipeline.spec import PipelineSpec, StageSpec


def test_pipeline_spec_stage_names_preserves_order():
    spec = PipelineSpec(
        name="default",
        stages=(
            StageSpec(name="retrieve"),
            StageSpec(name="fuse", config={"strategy_weights": {"semantic": 0.6}}),
            StageSpec(name="dedup_rerank"),
        ),
    )

    assert spec.stage_names() == ("retrieve", "fuse", "dedup_rerank")


def test_pipeline_spec_defaults_to_empty_stages():
    spec = PipelineSpec(name="empty")

    assert spec.stages == ()
    assert spec.stage_names() == ()


def test_stage_spec_config_defaults_to_empty_dict():
    stage = StageSpec(name="retrieve")

    assert stage.config == {}
