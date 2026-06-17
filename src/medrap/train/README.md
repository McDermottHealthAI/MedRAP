# medrap.train

PyTorch Lightning training infrastructure for the `medrap train` command. Contains the
supervised Lightning module wrapper, task and loss definitions (including marginalized
retrieval losses), AUROC and other scalar metrics, retrieval diagnostics logging, training
callbacks, and the Lightning datamodules used to feed data during training.
