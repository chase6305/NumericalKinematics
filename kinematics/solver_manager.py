import typing

from solver import Solver
from pinocchio_solver import PinocchioSolver
from pytorch_solver import PytorchSolver

__all__ = ["SolverManager"]


class SolverManager:
    def __init__(self):
        r"""Initializes the SolverManager with a dictionary of solver instances.
        The keys are the names of the solvers, and the values are the solver classes.
        """
        self.instances = {
            "PinocchioSolver": PinocchioSolver,
            "PytorchSolver": PytorchSolver,
        }

    def get_instance(self, solver_name: str) -> Solver:
        r"""Gets an instance of the specified solver class.

        Args:
            solver_name (str): The name of the solver to retrieve.

        Returns:
            type: The class of the solver if found, otherwise None.
        """
        return self.instances.get(solver_name)

    def get_all_instance(self) -> typing.List:
        r"""Gets a list of all solver classes managed by this manager.

        Returns:
            list: A list of all solver classes.
        """
        return list(self.instances.values())
