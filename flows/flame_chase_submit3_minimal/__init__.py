"""Submission-capped Flame Chase with trusted, container-isolated handoffs.

The whole experiment runs via ``python -m ...supervisor --config ...`` on the
Docker host, NOT inside an actor's writable container. ``actor_turn.py`` is the
single-session Humanize entry point used by that supervisor.
"""
