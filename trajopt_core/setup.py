import os
from glob import glob

from setuptools import find_packages, setup

package_name = "trajopt_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools", "numpy", "scipy", "casadi"],
    zip_safe=True,
    maintainer="Francesco Pedrini",
    maintainer_email="franci.pedrini@gmail.com",
    description=(
        "Platform-agnostic local trajectory optimisation: Gaussian occupancy grid, "
        "rolling-horizon A* and a CasADi/IPOPT receding-horizon OCP, instantiated on "
        "an aerial and a legged robot through a common MotionModel abstraction."
    ),
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": []},
)
