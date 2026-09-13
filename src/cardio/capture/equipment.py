"""Who made a capture, in the terms a DICOM receiver is told it.

The General Equipment module is how an archive says where an instance came
from, and for a derived series that is this app on somebody's workstation
rather than the scanner the pixels were acquired on.  None of it can be
discovered -- an institution is a fact about a deployment, not about a volume
-- so it is configured, and the lengths the value representations allow are
enforced on the fields rather than at the point of writing, the way
``Series.description`` already is.

The software version is deliberately not configurable: it is read off the
installed package, because a version anyone can type is a version that will
eventually disagree with the code that wrote the file -- which is the one
thing it exists to establish.
"""

# System
from importlib.metadata import version

# Third Party
import pydantic as pc

# What Long String and Short String hold.  A receiver rejects an overlong
# value rather than shortening it, so the limit belongs on the field.
LONG_STRING = 64
SHORT_STRING = 16


class Equipment(pc.BaseModel):
    """The equipment every instance of a capture says it came off.

    Extras are forbidden, as they are on ``Scene`` and ``Series``: a misspelled
    key in a hand-written config should say so rather than quietly do nothing.
    """

    model_config = pc.ConfigDict(extra="forbid")

    manufacturer: str = pc.Field(
        default="cardio",
        max_length=LONG_STRING,
        description="Manufacturer every written instance names.",
    )
    model_name: str = pc.Field(
        default="cardio",
        max_length=LONG_STRING,
        description="ManufacturerModelName every written instance names.",
    )
    institution_name: str = pc.Field(
        default="",
        max_length=LONG_STRING,
        description=(
            "InstitutionName every written instance names; empty writes none, "
            "which a research capture is warned about."
        ),
    )
    station_name: str = pc.Field(
        default="",
        max_length=SHORT_STRING,
        description="StationName every written instance names; empty writes none.",
    )
    device_serial_number: str = pc.Field(
        default="",
        max_length=LONG_STRING,
        description=(
            "DeviceSerialNumber every written instance names. Together with the "
            "software version this is what identifies the installation that "
            "produced a file, which is what a regulated deployment has to show."
        ),
    )

    @property
    def software_versions(self) -> str:
        """The installed package's version, which is not anyone's to set."""
        return version("cardio")

    @property
    def attributes(self) -> dict[str, str]:
        """The General Equipment arguments highdicom's constructors take.

        An unset name is left out rather than written empty: these are Type 3,
        and an absent attribute says "not recorded" where an empty one claims
        the value itself is blank.
        """
        fields = {
            "manufacturer": self.manufacturer,
            "manufacturer_model_name": self.model_name,
            "software_versions": self.software_versions,
            "device_serial_number": self.device_serial_number,
            "institution_name": self.institution_name,
        }
        return {name: value for name, value in fields.items() if value}
