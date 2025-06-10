import importlib
import logging
import yaml
from pathlib import Path
from impl import config as cfg

logger = logging.getLogger("impl")


class DomainFactory:
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
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

        try:
            config = self._get_component("Config")
            config_path = self.domain_path / config
            if config_path.is_file():
                domain_config = yaml.safe_load(config_path.read_text())
                cfg.CONFIG = cfg.merge_dict(cfg.CONFIG, domain_config)
            else:
                logger.warning(f"Domain config path does not exist: '{config_path}'.")
        except KeyError:
            logger.warning(f"Key 'Config' not found in the domain registry for '{domain_name}'.")

    def _get_component(self, name: str):
        try:
            return self.domain_dict[name]
        except KeyError:
            raise KeyError(
                f"Key '{name}' not found in the domain registry for '{self.domain_name}'.")

    def get_scenario(self):
        return self._get_component("Scenario")

    def get_scenario_init(self):
        return self._get_component("ScenarioInit")

    def get_perturbation_factory(self):
        return self._get_component("PerturbationFactory")

    def get_perturbation(self):
        return self._get_component("Perturbation")

    def get_evaluation(self):
        return self._get_component("Evaluation")

    def get_mr_set(self):
        return self._get_component("MRSet")
