"""Datasets by name. Every loader returns a polypbench.data.base.Dataset."""


def load_dataset(name):
    if name == "inhouse":
        from polypbench.data.inhouse import load_inhouse
        return load_inhouse()
    if name.startswith("sunseg_"):
        from polypbench.data.sunseg import SPLITS, load_sunseg
        if name in SPLITS:
            return load_sunseg(name)
    if name.startswith("realcolon_"):
        from polypbench.data.realcolon import SPLITS, load_realcolon
        if name in SPLITS:
            return load_realcolon(name)
    raise ValueError(f"Unknown dataset '{name}'")
