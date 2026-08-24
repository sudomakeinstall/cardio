"""Shared vue expressions for the drawer panels."""

# The drawer's MPR controls only make sense in the quad view with a volume
# selected. Written out nineteen times before this constant existed.
MPR_ACTIVE = "!maximized_view && active_volume_label"
