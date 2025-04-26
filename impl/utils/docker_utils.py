import docker
from docker.errors import NotFound
import subprocess
from impl.config import CONFIG
import time
import os
import logging

logger = logging.getLogger(__name__)

if CONFIG["simulation"]["docker"]["enabled"]:
    docker_client = docker.from_env()


def get_container(container_name):
    """
    Retrieve a running Docker container by its name.

    :param container_name: Name of the container.
    :return: Docker container object if running, otherwise :data:`None`.
    """
    try:
        container = docker_client.containers.get(container_name)
        if container.status != 'running':
            return None
        return container
    except NotFound:
        return None


def setup_carla(container_name, port):
    """
    Set up and run a CARLA server inside a Docker container.

    If the container is not running, a new one is launched with the correct settings.

    :param container_name: Desired name for the Docker container.
    :param port: World port to expose for CARLA server.
    :param gpu_device: GPU device index to assign to the container.
    :return: Docker container object.
    """
    carla_container = get_container(container_name)
    if carla_container is None or carla_container.status != 'running':
        logger.info("Setting up Carla...")
        default_conf = CONFIG["simulation"]["docker"]
        container_conf = [conf for conf in default_conf["instances"] \
                          if int(conf["port"]) == int(port)][0]

        process = subprocess.Popen([
            f"docker run --user {os.getuid()}:{os.getgid()} --privileged --rm --net=host" \
            + f" --memory {container_conf.get('memory', default_conf['memory'])}" \
            + f" --shm-size {container_conf.get('shared_memory', default_conf['shared_memory'])}" \
            + f" --cpus {container_conf.get('cpu', default_conf['cpu'])}" \
            + f" --gpus '\"device={container_conf['gpu_device']}\"'" \
            + f" --name {container_name} {CONFIG['simulation']['docker']['image']} ./CarlaUE4.sh --world-port={port}" \
            + f" --quality-level={'Epic' if CONFIG['simulation']['high_graphics'] else 'Low'} -opengl"
        ],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(15)
        carla_container = get_container(container_name)
    return carla_container


def restart_carla(container_name, port):
    """
    Restart the CARLA server inside a Docker container.

    :param container_name: Desired name for the Docker container.
    :param port: World port to expose for CARLA server.
    """
    logger.info("Stopping Carla container to restart...")
    process = subprocess.run([
        f"docker container stop {container_name}"
    ],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        time.sleep(5)
        setup_carla(container_name, port)
    else:
        raise Exception(
            f"There was an error while stopping container {container_name}: {process.stdout}, {process.stderr}"
        )


def user_has_processes_in_container(container_id, username):
    """
    Check whether a user has any active processes inside a given Docker container.

    :param container_id: ID of the Docker container.
    :param username: Username inside the container.
    :return: :data:`True` if processes are found, :data:`False` otherwise.
    """
    result = subprocess.run(['docker', 'exec', container_id, 'ps', '-u', username],
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def cleanup_containers():
    """
    Stop all Docker containers launched by the current user that match the CARLA Docker image.

    Only containers belonging to the current system user will be affected.
    """
    logger.info("Cleaning up Carla containers ...")
    container_ids = subprocess.run([
        f'docker ps -a --filter "name=^{CONFIG["simulation"]["docker"]["image"]}-" -q'],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip().split('\n')

    for container_id in container_ids:
        if container_id and user_has_processes_in_container(container_id, os.getlogin()):
            subprocess.run(['docker', 'stop', container_id])
            logger.info(f"Stopped container {container_id}.")
