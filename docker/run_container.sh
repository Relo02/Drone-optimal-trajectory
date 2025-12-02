# Fix permissions from host
docker-compose exec --user root drone-control chown -R user:user /opt/px4_source

# Then re-enter the container
docker-compose exec drone-control bash