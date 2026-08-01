"""
Entry point for the Message Notification Router.

Orchestrates the full pipeline: loads all dataset/*.csv files via loaders.py,
resolves and inspects media via media.py, retrieves similar historical
messages and reaction patterns via retrieval.py, runs the injection/scam
guard via safety.py, routes each message to notify/digest/mute via router.py,
and writes the final predictions to dataset/output.csv.

Run from the terminal, e.g.:
    python code/main.py
"""
