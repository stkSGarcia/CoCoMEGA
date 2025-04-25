class InvalidScenarioDefinitionError(Exception):
    """Exception raised for invalid scenario configuration."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadingScenarioFailedError(Exception):
    """Exception raised for failure of loading scenario."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class AgentSetupFailedError(Exception):
    """Exception raised for failure of setting up agent."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SimulationError(Exception):
    """Exception raised for error during the simulation."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class StoppingScenarioFailedError(Exception):
    """Exception raised for failure of stopping scenario."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class EarlyTerminationException(Exception):
    """Exception raised for early termination of scenario."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class AgentTerminationSignal(Exception):
    """Exception raised if agent terminates the scenario."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
