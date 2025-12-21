from setuptools import find_packages, setup

package_name = 'drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/laser_bridge.launch.py']),
        ('share/' + package_name + '/launch', ['launch/x500_state_publisher.launch.py']),
        ('share/' + package_name + '/urdf', ['urdf/x500_depth.urdf']),
        ('share/' + package_name + '/config', ['config/x500_depth_viz.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bridge = drone.bridge:main',
            'mission_commander = drone.mission_commander:main',
            'simple_px4_commander = drone.simple_px4_commander:main',
        ],
    },
)
