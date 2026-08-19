"""
Thin ROS 2 layer over trajopt_core.

The nodes contain no algorithm: they translate ROS messages into the core's data
structures and back.  Every topic name is relative, so a launch file decides what
robot the node is talking to.
"""
