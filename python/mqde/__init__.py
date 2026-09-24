"""MQDEsys Python executor: PyTorch-side quantization, evaluation and error tracing.

The C++ core (``mqde._core``) owns numerical semantics, model inspection and resource estimation;
this package drives PyTorch models under that contract.
"""

__version__ = "0.1.0"
