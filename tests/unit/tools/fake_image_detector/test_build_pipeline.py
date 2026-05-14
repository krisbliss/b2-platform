from tools.fake_image_detector.pipeline import FakeImageDetectorPipeline, build_pipeline


def test_build_pipeline_returns_pipeline_instance():
    pipeline = build_pipeline()
    assert isinstance(pipeline, FakeImageDetectorPipeline)


def test_build_pipeline_loads_at_least_one_check():
    pipeline = build_pipeline()
    assert len(pipeline._checks) > 0, (
        "build_pipeline() produced no checks — likely a silent import failure in one or more check modules"
    )


def test_build_pipeline_check_ids_are_strings():
    pipeline = build_pipeline()
    for cfg, check in pipeline._checks:
        assert isinstance(cfg.id, str) and cfg.id, f"check config has empty id: {cfg!r}"
