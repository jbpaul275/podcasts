class PipelineError(Exception):
    """A stage failed in a way that should fail the episode with a clear message."""


class DuplicateEpisode(PipelineError):
    """This PDF's content hash already has an episode. Carries that episode's
    id so the caller can point at it rather than reporting a bare failure."""

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        super().__init__(f"already ingested as episode {episode_id}")


class NoAudioError(PipelineError):
    """TTS returned text tokens instead of audio. Documented and intermittent,
    so it is worth retrying rather than failing the chunk outright."""


class QuotaUnavailable(PipelineError):
    """The plan has no allowance for this model at all (limit: 0), as opposed
    to being temporarily rate limited. Retrying can never succeed."""
