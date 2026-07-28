"""janus -- a video world-model with two faces.

One looks forward: the inferencer, serving predictions about the next half second.
One looks back: the learner, replaying what already happened. They share a head --
the same weights, published live from one face to the other while the stream runs.
"""

__version__ = "0.1.0"
