import typing

import pydantic as pc

type ScalarComponent = typing.Annotated[
    float, pc.Field(ge=0.0, le=1.0, validate_default=True)
]

type RGBColor = typing.Annotated[
    tuple[
        ScalarComponent,
        ScalarComponent,
        ScalarComponent,
    ],
    pc.Field(validate_default=True),
]
