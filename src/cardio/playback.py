"""How the cine plays: its speed, its direction, and what it costs to send."""

# Third Party
import pydantic as pc

# Internal
from .image_quality import DEFAULT_PLAYBACK_QUALITY, DEFAULT_PLAYBACK_RESOLUTION


class Playback(pc.BaseModel):
    """The playback controls' starting positions.

    ``bpm`` and ``bpr`` describe the acquisition rather than a preference, so
    they belong with the dataset. The bounds match the sliders that set them.

    Extras are forbidden, as they are on ``Scene``: a misspelled key in a
    hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    bpm: int = pc.Field(
        default=60, ge=20, le=120, description="Playback speed, in beats per minute"
    )
    bpr: int = pc.Field(
        default=3,
        ge=1,
        le=360,
        description="Cardiac cycles per full rotation of the camera",
    )
    incrementing: bool = pc.Field(
        default=True, description="Advance frames while playing"
    )
    rotating: bool = pc.Field(
        default=False, description="Rotate the camera while playing"
    )
    quality: int = pc.Field(
        default=DEFAULT_PLAYBACK_QUALITY,
        ge=10,
        le=100,
        description="JPEG encode quality while playing; 100 is full quality",
    )
    resolution: int = pc.Field(
        default=DEFAULT_PLAYBACK_RESOLUTION,
        ge=25,
        le=100,
        description="Render resolution while playing, as a percent of full",
    )
