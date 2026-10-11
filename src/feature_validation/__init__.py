"""Post-hoc validation of generated gait features.

This package deliberately sits outside the extraction pipeline. It reads features
that have already been written to disk and reports whether their values are
*finite* (present rather than missing) and *meaningful* (within physiological
range), without gating or altering extraction output.
"""
