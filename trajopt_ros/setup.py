import os
from glob import glob

from setuptools import find_packages, setup

package_name = "trajopt_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Francesco Pedrini",
    maintainer_email="franci.pedrini@gmail.com",
    description="ROS 2 wrappers for trajopt_core (platform-independent A* + MPC nodes)",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "a_star_node = trajopt_ros.a_star_node:main",
            "mpc_node = trajopt_ros.mpc_node:main",
        ],
    },
)
