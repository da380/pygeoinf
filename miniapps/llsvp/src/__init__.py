"""The LLSVP app's modules: the shell and its network, the prior, the forward map, the helpers."""

import os

# The sphere's transforms read this once, when they are first imported. At these
# sizes one thread is several times faster than one per core, which spend their
# time waiting on each other. A value already in the environment is respected.
os.environ.setdefault("OMP_NUM_THREADS", "1")
