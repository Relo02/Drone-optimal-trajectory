import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mujoco_sim'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lorenzo',
    maintainer_email='lorenzo1.ortolani@mail.polimi.it',
    description='MuJoCo drone simulation with A* path planning',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'a_star_planner_node = mujoco_sim.a_star_planner_node:main',
            'mujoco_sim_node = mujoco_sim.main:main',
            'pointcloud_visualizer_node = mujoco_sim.visualize_pointcloud:main',
        ],
    },
)
