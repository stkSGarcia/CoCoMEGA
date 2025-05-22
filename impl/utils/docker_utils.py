import docker
from docker.errors import NotFound
import subprocess
from impl import config as cfg
import time
import os
import logging

logger = logging.getLogger(__name__)

if cfg.CONFIG["docker"]["enabled"]:
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


def setup_carla(container_name, port, gpu_device):
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
        default_conf = cfg.CONFIG["docker"]

        process = subprocess.Popen([
            f"docker run --user {os.getuid()}:{os.getgid()} --privileged --rm --net=host" \
            + f" --memory {default_conf['memory']}" \
            + f" --shm-size {default_conf['shared_memory']}" \
            + f" --cpus {default_conf['cpu']}" \
            + f" --gpus '\"device={gpu_device}\"'" \
            + f" --name {container_name} {cfg.CONFIG['docker']['image']} ./CarlaUE4.sh --world-port={port}" \
            + f" --quality-level={'Epic' if cfg.CONFIG['simulation']['high_graphics'] else 'Low'} -opengl"
        ],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(15)
        carla_container = get_container(container_name)
    return carla_container


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


def cleanup_containers(tag=None):
    """
    Stop all Docker containers launched by the current user that match the CARLA Docker image.

    Only containers belonging to the current system user will be affected.
    """
    pattern = f'{cfg.CONFIG["docker"]["image"]}-{tag if tag is not None else ""}'
    logger.info("Cleaning up Carla containers ...")
    command = f'docker ps -a --filter "name=^{pattern}" -q | while read container; do ' \
              'pid=$(docker inspect --format "{{{{ .State.Pid }}}}" "$container" 2>/dev/null); ' \
              'if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then ' \
              'owner_uid=$(stat -c %u /proc/$pid/); ' \
              'if [ "$owner_uid" -eq "$UID" ]; then ' \
              'docker stop "$container"; ' \
              'fi; fi; done'
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
