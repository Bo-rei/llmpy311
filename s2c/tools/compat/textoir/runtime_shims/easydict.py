"""Minimal EasyDict compatibility shim for the isolated TextOIR runner.

The upstream TextOIR code only needs dictionary values to be accessible through
attributes.  Keeping this shim inside the compatibility layer avoids mutating
the project's Python environment or silently installing a new dependency.
"""

from __future__ import annotations


__s2c_shim__ = True


class EasyDict(dict):
    """A small drop-in subset of ``easydict.EasyDict`` used by TextOIR."""

    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def update(self, mapping=None, **kwargs):
        if mapping is not None:
            items = mapping.items() if hasattr(mapping, "items") else mapping
            for key, value in items:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value
