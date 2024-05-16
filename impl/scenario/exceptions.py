class InvalidScenarioDefinitionError(Exception):
    """Exception raised for invalid scenario configuration.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadingScenarioFailedError(Exception):
    """Exception raised for failure of loading scenario.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class AgentSetupFailedError(Exception):
    """Exception raised for failure of setting up agent.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SimulationError(Exception):
    """Exception raised for error during the simulation.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class StoppingScenarioFailedError(Exception):
    """Exception raised for failure of stopping scenario.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class EarlyTerminationException(Exception):
    """Exception raised for early termination of scenario.

        Attributes:
            message -- explanation of the error
        """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
