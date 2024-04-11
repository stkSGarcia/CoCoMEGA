import docker
from docker.errors import NotFound
import subprocess
from impl.config import CONFIG
import time
import logging

logger = logging.getLogger(__name__)
docker_client = docker.from_env()


def get_container(container_name):
    try:
        container = docker_client.containers.get(container_name)
        if container.status != 'running':
            return None
        return container
    except NotFound:
        return None


def setup_carla(container_name, port):
    carla_container = get_container(container_name)
    if carla_container is None or carla_container.status != 'running':
        logger.info("Setting up Carla...")
        container_conf = [conf for conf in CONFIG["simulation"]["docker"]["instances"] \
                          if int(conf["port"]) == int(port)][0]
        process = subprocess.Popen([
            f"docker run --privileged --rm --net=host --memory {container_conf['memory']}" \
            + f" --cpus {container_conf['cpu']} --gpus '\"device={container_conf['gpu_device']}\"'" \
            + f" --name {container_name} {CONFIG['simulation']['docker']['image']} ./CarlaUE4.sh --world-port={port} -opengl"
        ],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(10)
        carla_container = get_container(container_name)
    return carla_container


def restart_carla(container_name, port):
    logger.info("Stopping Carla container to restart...")
    process = subprocess.run([
        f"docker stop {container_name}"
    ],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        logger.info("Setting up Carla...")
        setup_carla(container_name, port)
    else:
        raise Exception(
            f"There was an error while stopping container {container_name}: {process.stdout}, {process.stderr}"
        )
