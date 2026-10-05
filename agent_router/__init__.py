"""Jev router: typed classification in front of the orchestrator.

The router asks Jev (TypeSafe System One) a fixed set of atomic questions and
turns the typed answers into a RoutingPlan with plain code. No LLM interprets
the plan, and Jev never gets write, shell or tool access.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
