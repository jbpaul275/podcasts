class PipelineError(Exception):
    """A stage failed in a way that should fail the episode with a clear message."""


class NoAudioError(PipelineError):
    """TTS returned text tokens instead of audio. Documented and intermittent,
    so it is worth retrying rather than failing the chunk outright."""


class QuotaUnavailable(PipelineError):
    """The plan has no allowance for this model at all (limit: 0), as opposed
    to being temporarily rate limited. Retrying can never succeed."""
