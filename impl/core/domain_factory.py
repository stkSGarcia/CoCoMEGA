import importlib
import logging
from pathlib import Path

import yaml

from impl import config as cfg

logger = logging.getLogger(__name__)


class DomainFactory:
    """
    Factory class for loading domain-specific components.
    This class dynamically imports the domain registry module and retrieves
    the components defined in the :const:`DOMAIN_REGISTRY` dictionary.
    It also handles the loading of domain-specific configuration files.

    Attributes:
        domain_name (str): Name of the domain to load.
        domain_registry (module): The imported module for the domain registry.
        domain_path (:class:`pathlib.Path`): Path to the domain registry module.
        domain_dict (dict): Dictionary containing the domain components.

    Raises:
        ImportError: If the domain registry module cannot be found.
        AttributeError: If the :const:`DOMAIN_REGISTRY` attribute is not found in the module.
        TypeError: If the :const:`DOMAIN_REGISTRY` is not a dictionary.
    """

    def __init__(self, domain_name: str):
        """
        Initializes the :class:`DomainFactory` with the specified domain name.
        """
        self.domain_name = domain_name

        try:
            domain_config_path = Path(f"impl/{domain_name}") / "config.yaml"
            custom_config_path = Path("config.yaml")
            domain_config = yaml.safe_load(domain_config_path.read_text()) if domain_config_path.is_file() else {}
            custom_config = yaml.safe_load(custom_config_path.read_text()) if custom_config_path.is_file() else {}

            cfg.merge_dict(cfg.CONFIG, domain_config, exclude=["workspace"])
            cfg.merge_dict(cfg.CONFIG, custom_config, exclude=["workspace"])
        except Exception as e:
            logger.warning(f"Exception Occurred While Reading Config: {e}")

        try:
            self.domain_registry = importlib.import_module(f"impl.{domain_name}.register")
            self.domain_path = Path(self.domain_registry.__file__).parent
        except ModuleNotFoundError as e:
            raise ImportError(f"Domain {domain_name} not found") from e

        try:
            self.domain_dict = getattr(self.domain_registry, "DOMAIN_REGISTRY")
        except AttributeError:
            raise AttributeError(f"'DOMAIN_REGISTRY' not found in 'domains.{domain_name}.register'.")

        if not isinstance(self.domain_dict, dict):
            raise TypeError(f"'DOMAIN_REGISTRY' in 'domains.{domain_name}.register' must be a dict.")

    def _get_component(self, name: str):
        try:
            return self.domain_dict[name]
        except KeyError:
            raise KeyError(
                f"Key '{name}' not found in the domain registry for '{self.domain_name}'.")

    def get_scenario(self):
        """
        Retrieves the scenario component from the domain registry.
        """
        return self._get_component("Scenario")

    def get_scenario_init(self):
        """
        Retrieves the scenario initialization component from the domain registry.
        """
        return self._get_component("ScenarioInit")

    def get_perturbation_factory(self):
        """
        Retrieves the perturbation factory component from the domain registry.
        """
        return self._get_component("PerturbationFactory")

    def get_perturbation(self):
        """
        Retrieves the perturbation component from the domain registry.
        """
        return self._get_component("Perturbation")

    def get_perturbation_init(self):
        """
        Retrieves the perturbation initialization component from the domain registry.
        """
        return self._get_component("PerturbationInit")

    def get_evaluation(self):
        """
        Retrieves the evaluation component from the domain registry.
        """
        return self._get_component("Evaluation")

    def get_mr_set(self):
        """
        Retrieves the MRSet component from the domain registry.
        """
        return self._get_component("MRSet")
