"""
Configuration for brain connection management.

Uses UDP multicast discovery - no static peer configuration needed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BrainConnectionConfig:
    """
    Configuration for brain network connections.

    Uses UDP multicast discovery - peers are discovered automatically.
    """
    brain_id: str
    listen_address: Optional[Tuple[str, int]] = None
    enable_listener: bool = True
    authkey: bytes = b"brain-secret"

    def __post_init__(self):
        """Set defaults after initialization."""
        # Auto-assign listen address if listener enabled but no address specified
        if self.enable_listener and self.listen_address is None:
            import socket
            # Find an available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', 0))
                _, port = s.getsockname()
                self.listen_address = ('0.0.0.0', port)

    @classmethod
    def from_args(
        cls,
        host: str = "localhost",  # Ignored - kept for backwards compatibility
        port: int = 6000,  # Ignored - kept for backwards compatibility
        authkey: bytes = b"brain-secret",
        brain_id: Optional[str] = None,
        listen_host: Optional[str] = None,
        listen_port: Optional[int] = None,
        enable_listener: bool = True,
    ) -> "BrainConnectionConfig":
        """
        Create config from command-line arguments.

        Note: host/port are ignored - peers are discovered via multicast.
        """
        # Generate brain_id if not provided
        if brain_id is None:
            import uuid
            brain_id = f"brain_{uuid.uuid4().hex[:8]}"

        # Set up listener address
        listen_address = None
        if enable_listener:
            if listen_host is not None and listen_port is not None:
                listen_address = (listen_host, listen_port)
            elif listen_host is not None:
                raise ValueError(
                    "--listen-port required when --listen-host is specified")
            else:
                # Auto-assign port if not specified
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', 0))
                    _, auto_port = s.getsockname()
                    listen_address = ("0.0.0.0", auto_port)

        return cls(
            brain_id=brain_id,
            listen_address=listen_address,
            enable_listener=enable_listener,
            authkey=authkey,
        )
