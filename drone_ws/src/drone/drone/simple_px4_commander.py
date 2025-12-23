#!/usr/bin/env python3
# find_params.py

from pymavlink import mavutil
import time

def find_params():
    # Connect to PX4
    print("Connecting to PX4...")
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    
    # Wait for heartbeat
    master.wait_heartbeat()
    print("Connected!")
    
    # Request parameter list
    master.mav.param_request_list_send(
        master.target_system,
        master.target_component
    )
    
    print("Requested parameter list. Checking common parameters...")
    
    # Common parameter patterns to check
    patterns = [
        'COM_DL', 'COM_ARM', 'CBRK', 'SAFETY', 
        'NAV_RCL', 'COM_RC', 'COM_OF', 'COM_GCS',
        'MAV_', 'LINK_', 'ARM_', 'DISARM_'
    ]
    
    # Listen for parameters
    params_found = {}
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Listen for 10 seconds
        msg = master.recv_match(type=['PARAM_VALUE'], blocking=True, timeout=1)
        if msg:
            param_name = msg.param_id.decode('utf-8').strip('\x00')
            
            # Check if it matches any pattern
            for pattern in patterns:
                if pattern in param_name:
                    params_found[param_name] = msg.param_value
                    print(f"Found: {param_name} = {msg.param_value}")
    
    print("\n" + "="*60)
    print("RELEVANT PARAMETERS FOUND:")
    print("="*60)
    
    for name, value in params_found.items():
        print(f"{name:30} = {value}")
    
    return params_found

if __name__ == '__main__':
    find_params()