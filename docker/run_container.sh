# Fix permissions
docker exec --user root drone_control_container chown -R user:user /PX4-Autopilot

# Enter container
docker exec -it drone_control_container bash