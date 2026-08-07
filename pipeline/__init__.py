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


class ModelUnusable(PipelineError):
    """This model cannot serve this request, and no amount of retrying changes
    that. The useful response is to try a different model, so callers with a
    fallback catch this rather than the specific reasons below."""


class QuotaUnavailable(ModelUnusable):
    """The plan has no allowance for this model at all (limit: 0), as opposed
    to being temporarily rate limited. Retrying can never succeed."""


class ModelRetired(ModelUnusable):
    """The model ID 404s: withdrawn, renamed, or wrong. Providers retire
    preview models on their own schedule, so a config that worked last week
    can start failing without anything here changing."""
