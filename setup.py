from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="cg",
    version="1.0",
    packages=["cg"],
    rust_extensions=[
        RustExtension(
            "cg_rustpy._cg_rustpy", binding=Binding.PyO3, path="rust/Cargo.toml"
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
