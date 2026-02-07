#!/usr/bin/env python3
"""Generate cf2.xml with 360-degree lidar coverage."""

import math

# Configuration
NUM_LIDAR_RAYS = 360  # 1 degree resolution
LIDAR_ANGLE_STEP = 2 * math.pi / NUM_LIDAR_RAYS

def generate_lidar_sites():
    """Generate lidar site definitions."""
    sites = []
    for i in range(NUM_LIDAR_RAYS):
        angle = i * LIDAR_ANGLE_STEP
        # Quaternion to rotate +Z to point outward at angle θ in XY plane
        # Rotation axis perpendicular to (0,0,1) and (cos(θ), sin(θ), 0)
        # is (-sin(θ), cos(θ), 0), rotation angle is π/2
        # q = (cos(π/4), sin(π/4)*axis) = (0.7071, -0.7071*sin(θ), 0.7071*cos(θ), 0)
        c = math.cos(angle)
        s = math.sin(angle)
        qw = 0.7071067811865476
        qx = -0.7071067811865476 * s
        qy = 0.7071067811865476 * c
        qz = 0.0
        sites.append(f'      <site name="lidar_{i}" size="0.001" pos="0 0 0" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"/>')
    return '\n'.join(sites)

def generate_lidar_sensors():
    """Generate rangefinder sensor definitions."""
    sensors = []
    for i in range(NUM_LIDAR_RAYS):
        # cutoff=20 means 20 meter max range
        sensors.append(f'    <rangefinder name="lidar_{i}" site="lidar_{i}" cutoff="20"/>')
    return '\n'.join(sensors)

def generate_collision_meshes():
    """Generate collision mesh asset and geom definitions."""
    assets = []
    geoms = []
    for i in range(32):
        assets.append(f'    <mesh file="cf2_collision_{i}.obj"/>')
        geoms.append(f'      <geom class="collision" mesh="cf2_collision_{i}"/>')
    return '\n'.join(assets), '\n'.join(geoms)

collision_assets, collision_geoms = generate_collision_meshes()

xml_content = f'''<?xml version="1.0"?>
<mujoco model="cf2">
  <compiler coordinate="local" meshdir="cf2" angle="radian"/>

  <option integrator="RK4"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
      <default class="collision">
        <geom group="3" type="mesh"/>
      </default>
    </default>
  </default>

  <asset>
    <!-- Materials for CF2 drone -->
    <material name="Material10-polished_plastic1_001" specular="0.5" shininess="0.225"
      rgba="0.631373 0.658824 0.678431 1.0"/>
    <material name="Material12-polished_gold_001" specular="0.5" shininess="0.225"
      rgba="0.968627 0.878431 0.600000 1.0"/>
    <material name="Material14-green_medium_gloss_plastic_001" specular="0.5" shininess="0.225"
      rgba="0.109804 0.184314 0.000000 1.0"/>
    <material name="Material15_001" specular="0.5" shininess="0.225"
      rgba="0.792157 0.819608 0.933333 1.0"/>
    <material name="Material3-polypropylene2_001" specular="0.5" shininess="0.225"
      rgba="1.0 1.0 1.0 1.0"/>
    <material name="Material6-black_medium_gloss_plastic_001" specular="0.5" shininess="0.225"
      rgba="0.101961 0.101961 0.101961 1.0"/>
    <material name="Material8-burnished_chrome_001" specular="0.5" shininess="0.225"
      rgba="0.898039 0.898039 0.898039 1.0"/>

    <!-- Visual meshes -->
    <mesh file="cf2_0.obj"/>
    <mesh file="cf2_1.obj"/>
    <mesh file="cf2_2.obj"/>
    <mesh file="cf2_3.obj"/>
    <mesh file="cf2_4.obj"/>
    <mesh file="cf2_5.obj"/>
    <mesh file="cf2_6.obj"/>

    <!-- Collision meshes -->
{collision_assets}
  </asset>

  <worldbody>
    <body name="cf2" pos="0 0 1.0">
      <inertial pos="0 0 0" mass="0.027" diaginertia="2.3951e-05 2.3951e-05 3.2347e-05"/>
      <joint name="root" type="free" damping="0" armature="0"/>

      <!-- Visual geometries -->
      <geom class="visual" mesh="cf2_0" material="Material10-polished_plastic1_001"/>
      <geom class="visual" mesh="cf2_1" material="Material12-polished_gold_001"/>
      <geom class="visual" mesh="cf2_2" material="Material14-green_medium_gloss_plastic_001"/>
      <geom class="visual" mesh="cf2_3" material="Material15_001"/>
      <geom class="visual" mesh="cf2_4" material="Material3-polypropylene2_001"/>
      <geom class="visual" mesh="cf2_5" material="Material6-black_medium_gloss_plastic_001"/>
      <geom class="visual" mesh="cf2_6" material="Material8-burnished_chrome_001"/>

      <!-- Collision geometries -->
{collision_geoms}

      <!-- Actuation site at center of mass -->
      <site name="actuation" type="sphere" size="0.001" rgba="1 0 0 1" pos="0 0 0"/>

      <!-- IMU sensor site -->
      <site name="imu" type="box" size="0.002 0.002 0.002" rgba="0 1 0 1" pos="0 0 0"/>

      <!-- LIDAR: {NUM_LIDAR_RAYS} rays in horizontal plane (1 degree resolution) -->
{generate_lidar_sites()}
    </body>
  </worldbody>

  <actuator>
    <!-- Combined thrust and moment actuation -->
    <motor name="body_thrust" site="actuation" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="0 0.6"/>
    <motor name="x_moment" site="actuation" gear="0 0 0 1 0 0" ctrllimited="true" ctrlrange="-0.005 0.005"/>
    <motor name="y_moment" site="actuation" gear="0 0 0 0 1 0" ctrllimited="true" ctrlrange="-0.005 0.005"/>
    <motor name="z_moment" site="actuation" gear="0 0 0 0 0 1" ctrllimited="true" ctrlrange="-0.005 0.005"/>
  </actuator>

  <sensor>
    <!-- IMU sensors -->
    <accelerometer name="accel" site="imu"/>
    <gyro name="gyro" site="imu"/>

    <!-- Position and velocity sensors -->
    <framepos name="cf2_position" objtype="body" objname="cf2"/>
    <framequat name="cf2_orientation" objtype="body" objname="cf2"/>
    <framelinvel name="cf2_velocity" objtype="body" objname="cf2"/>
    <frameangvel name="cf2_angular_velocity" objtype="body" objname="cf2"/>

    <!-- LIDAR rangefinders ({NUM_LIDAR_RAYS} rays) -->
{generate_lidar_sensors()}
  </sensor>
</mujoco>
'''

if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'cf2.xml')

    with open(output_path, 'w') as f:
        f.write(xml_content)

    print(f'Generated {output_path}')
    print(f'  - {NUM_LIDAR_RAYS} lidar rays (1° resolution)')
    print(f'  - 32 collision meshes')
    print(f'  - 7 visual meshes')
