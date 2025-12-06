import typing

import pydantic as pc

ScalarComponent: typing.TypeAlias = typing.Annotated[
    float, pc.Field(ge=0.0, le=1.0, validate_default=True)
]

RGBColor: typing.TypeAlias = typing.Annotated[
    tuple[
        ScalarComponent,
        ScalarComponent,
        ScalarComponent,
    ],
    pc.Field(validate_default=True),
]
