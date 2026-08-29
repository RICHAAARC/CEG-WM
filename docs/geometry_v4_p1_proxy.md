# Geometry-V4 P1 RGB proxy

P1D-A is a local NumPy/Pillow-compatible ordinary-RGB proxy. The writer normalizes
the detection key once, derives only the Geometry HKDF key, and adds a keyed
multiscale spectral field plus fixed 4x4 tiles under a bounded RGB budget. The
detector accepts attacked RGB and key only; it estimates coarse translation by
phase correlation before emitting a coordinate-only, fail-closed record. P1D
is development evidence only; P1C is declared but not run here. Operators,
matching, aggregate and tie-break details are P1 development records, not P0.
