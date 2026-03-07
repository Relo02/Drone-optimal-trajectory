import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'new_mujoco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lorenzo',
    maintainer_email='lorenzo1.ortolani@mail.polimi.it',
    description='Skydio X2 MuJoCo simulation with cascaded PID and 3-D lidar',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'skydio_sim_node = new_mujoco.skydio_sim_node:main',
            'a_star_node = new_mujoco.a_star_node:main',
            'mpc_node = new_mujoco.mpc_node:main',
        ],
    },
)
