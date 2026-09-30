"""Datasets by name. Every loader returns a polypbench.data.base.Dataset."""


def load_dataset(name):
    if name == "inhouse":
        from polypbench.data.inhouse import load_inhouse
        return load_inhouse()
    raise ValueError(f"Unknown dataset '{name}'")
