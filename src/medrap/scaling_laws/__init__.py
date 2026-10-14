"""Scaling-law sweep: grid construction, dataset-size subsampling, results schema, and curve fitting.

Supports the experiment described in the project's scaling-laws design doc: fit
``y = A + B * x^-alpha`` independently over dataset size and model capacity, with and
without retrieval, and check whether the gap to the fitted asymptote predicts how much
retrieval helps.

This package only builds and analyzes the sweep; it does not submit or run training
jobs itself (see each module's docstring for what it assumes vs. what the caller
supplies, e.g. cluster-specific submission commands or exact retrieval-arm configs).
"""
