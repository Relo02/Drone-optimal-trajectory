import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'go2_mujoco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='francesco',
    maintainer_email='franci.pedrini@gmail.com',
    description='Unitree Go2 warehouse simulation in MuJoCo, driven by the shared trajopt stack',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'go2_sim_node = unitree_mujoco.robot_sim_node:main',
            'setpoint_to_cmd_vel_node = unitree_mujoco.setpoint_to_cmd_vel_node:main',
        ],
    },
)
