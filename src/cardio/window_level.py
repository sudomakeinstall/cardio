import functools as ft

import pydantic as pc


@pc.dataclasses.dataclass(config=dict(frozen=True))
class WindowLevel:
    name: str
    window: float
    level: float

    @pc.computed_field
    @ft.cached_property
    def lower(self) -> float:
        return self.level - self.window / 2

    @pc.computed_field
    @ft.cached_property
    def upper(self) -> float:
        return self.level + self.window / 2


presets = {
    1: WindowLevel("Abdomen", 400, 40),
    2: WindowLevel("Lung", 1500, -700),
    3: WindowLevel("Liver", 100, 110),
    4: WindowLevel("Bone", 1500, 500),
    5: WindowLevel("Brain", 85, 42),
    6: WindowLevel("Stroke", 36, 28),
    7: WindowLevel("Vascular", 800, 200),
    8: WindowLevel("Subdural", 160, 60),
    9: WindowLevel("Normalized", 0, 1),
}
