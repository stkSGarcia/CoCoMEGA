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
        process = subprocess.Popen([
            f"docker run --privileged --rm --net=host --memory {CONFIG['simulation']['docker']['memory']} --cpus {CONFIG['simulation']['docker']['cpu']} --gpus {CONFIG['simulation']['docker']['gpu']}" \
            + f" --name {container_name} {CONFIG['simulation']['docker']['image']} ./CarlaUE4.sh --world-port={port} -opengl"
        ],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("Setting up Carla...")
        time.sleep(10)
        # try:
        #     time.sleep(5)
        #     _, stderr = process.communicate(timeout=5)  # Adjust timeout as needed
        #     if stderr:
        #         logger.error("Carla error detected:", stderr)
        #         raise Exception('Error occurred while starting Carla.')
        # except subprocess.TimeoutExpired:
        #     pass

        # return_code = process.wait()
        # if return_code != 0:
        #     raise Exception(f"There was an error building Carla: {process.stdout}, {process.stderr}")

        carla_container = get_container(container_name)
    return carla_container


def restart_carla(container_name):
    process = subprocess.run([
        f"docker restart {container_name}"
    ],
        shell=True, stdout=subprocess.PIPE)
    if process.returncode != 0:
        raise Exception(f"There was an error while restarting Carla: {process.stdout}")
