# protocol.py
# Message types as plain string constants
OBSERVATION = "observation"
REWARD = "reward"
ACTION = "action"
TERMINAL = "terminal"
SHUTDOWN = "shutdown"

# Optional envelope fields (added by ConnectionManager for routing):
# - from_id: str - Identifier of the sending node
# - to_id: Optional[str] - Target peer identifier (None = broadcast)
# These fields are automatically handled by ConnectionManager and can be
# ignored by code that doesn't need multi-peer routing.
