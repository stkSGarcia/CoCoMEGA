import logging
import time

logger = logging.getLogger(__name__)


class Budget:
    def __init__(self, max_sim, max_time, max_gen):
        """Constructor.

        @param max_sim: The maximum number of simulations for the search.
        @param max_time: The maximum execution time for the search.
        @param max_gen: The maximum number of iterations for the search.
        """
        self.max_sim = max_sim
        self.max_time = max_time
        self.max_gen = max_gen
        self.sim_num = None
        self.start_time = None
        self.gen_num = None

    def initialize(self, other=None):
        if other and isinstance(other, self.__class__):
            self.sim_num = other.sim_num
            self.start_time = other.start_time
            self.gen_num = other.gen_num
        else:
            self.sim_num = 0
            self.start_time = time.perf_counter()
            self.gen_num = 0

    def acc_sim(self, n):
        self.sim_num += n

    def acc_gen(self):
        self.gen_num += 1

    def is_reached(self):
        """Determine if the budget is reached.

        @return: Return `True` if the budget is reached, `False` otherwise.
        """
        return ((self.max_sim is not None and self.sim_num > self.max_sim) or
                (self.max_time is not None and time.perf_counter() - self.start_time > self.max_time) or
                (self.max_gen is not None and self.gen_num > self.max_gen))

    def print_budget(self):
        return f"Budget: max simulations: {self.max_sim}, max time: {self.max_time}, max generations: {self.max_gen}."
