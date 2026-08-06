class PipelineError(Exception):
    """A stage failed in a way that should fail the episode with a clear message."""
