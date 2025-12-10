"""
Configuration for brain connection management.

Supports both incoming (Listener) and outgoing (Client) connections
while maintaining backwards compatibility with single-client mode.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class BrainConnectionConfig:
    """
    Configuration for brain network connections.

    Defaults ensure backwards compatibility with existing "brain as client" behavior.
    """
    brain_id: str
    listen_address: Optional[Tuple[str, int]] = None
    peers: List[Tuple[str, str, int]] = None  # List of (peer_id, host, port)
    enable_listener: bool = False
    authkey: bytes = b"brain-secret"
    default_peer_id: str = "env"

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.peers is None:
            self.peers = []

    @classmethod
    def from_args(
        cls,
        host: str = "localhost",
        port: int = 6000,
        authkey: bytes = b"brain-secret",
        brain_id: Optional[str] = None,
        listen_host: Optional[str] = None,
        listen_port: Optional[int] = None,
        peers: Optional[List[Tuple[str, str, int]]] = None,
        enable_listener: bool = False,
        default_peer_id: str = "env",
    ) -> "BrainConnectionConfig":
        """
        Create config from command-line arguments.

        Maintains backwards compatibility: if no new args provided,
        creates single-client config pointing to (host, port).
        """
        # Generate brain_id if not provided
        if brain_id is None:
            import uuid
            brain_id = f"brain_{uuid.uuid4().hex[:8]}"

        # Default peer connection (backwards compatibility)
        default_peers = [(default_peer_id, host, port)]

        # Merge with any additional peers
        if peers:
            # Check if default peer already exists
            peer_ids = {p[0] for p in peers}
            if default_peer_id not in peer_ids:
                final_peers = default_peers + peers
            else:
                final_peers = peers
        else:
            final_peers = default_peers

        # Set up listener address if enabled
        listen_address = None
        if enable_listener and listen_host is not None:
            if listen_port is None:
                raise ValueError(
                    "--listen-port required when --listen-host is specified")
            listen_address = (listen_host, listen_port)
        elif enable_listener:
            # Default listen address if enable_listener but no host specified
            # Default offset to avoid conflicts
            listen_address = ("localhost", port + 1000)

        return cls(
            brain_id=brain_id,
            listen_address=listen_address,
            peers=final_peers,
            enable_listener=enable_listener,
            authkey=authkey,
            default_peer_id=default_peer_id,
        )
