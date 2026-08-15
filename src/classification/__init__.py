"""Classifier stage of the drone imagery pipeline.

Takes an image crop produced by the SAM stage and assigns one of the target
classes (building, tree, car, other) with a confidence score.
"""
