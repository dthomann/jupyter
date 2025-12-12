# protocol.py
# Message types as plain string constants
OBSERVATION = "observation"
REWARD = "reward"
ACTION = "action"
TERMINAL = "terminal"
SHUTDOWN = "shutdown"

# Discovery protocol messages
DISCOVERY_STARTUP = "discovery/startup"        # Broadcast when peer starts up
DISCOVERY_SHUTDOWN = "discovery/shutdown"      # Broadcast when peer shuts down
DISCOVERY_ANNOUNCE = "discovery/announce"      # Reply to startup with peer info
DISCOVERY_LIST_PEERS = "discovery/list_peers"  # Request list of available peers
DISCOVERY_PEER_LIST = "discovery/peer_list"    # Response with list of peers

# Peer types
PEER_TYPE_BRAIN = "brain"
PEER_TYPE_ENVIRONMENT = "environment"

# Optional envelope fields (added by ConnectionManager for routing):
# - from_id: str - Identifier of the sending node
# - to_id: Optional[str] - Target peer identifier (None = broadcast)
# These fields are automatically handled by ConnectionManager and can be
# ignored by code that doesn't need multi-peer routing.
