"""Writing numbers down in the form DICOM's value representations allow.

A value representation is a width as much as a type: Decimal String holds
sixteen characters, and a float64 printed in full routinely needs more.  A
direction cosine off an oblique reformat prints twenty-one, which is every
reformat this app makes, cardiac planes being oblique by definition.

pydicom writes an overlong value without complaining and reads it back again,
so nothing here notices; the receiver is the one that has to decide what to do
with it, and it may refuse the instance or truncate the number and place the
image somewhere it never was.  ``decimal`` is what keeps that from being
something anyone has to remember at the point of writing.
"""

# Third Party
import pydicom as pd


def decimal(value: float) -> pd.valuerep.DSfloat:
    """One number as a Decimal String: the same value, shortened to fit.

    ``auto_format`` is pydicom's own, and is what highdicom reaches for
    internally, so a value this app writes and one highdicom writes are
    rounded the same way.
    """
    return pd.valuerep.DS(float(value), auto_format=True)


def decimals(values) -> list[pd.valuerep.DSfloat]:
    """A vector as Decimal Strings, for the attributes that carry several."""
    return [decimal(value) for value in values]
