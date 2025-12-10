"""
ConnectionManager for bi-directional brain communication.

Manages both incoming (Listener) and outgoing (Client) connections,
providing a unified interface for message passing.
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from multiprocessing.connection import Listener, Client, Connection
import select
import socket

from .connection_config import BrainConnectionConfig


class ConnectionManager:
    """
    Manages all connections for a brain process.

    Supports:
    - Zero or one Listener for incoming connections
    - Zero or more Client connections for outgoing connections
    - Non-blocking message polling
    - Automatic reconnection for outgoing peers
    - Message envelope handling (from_id, to_id)
    """

    def __init__(self, config: BrainConnectionConfig):
        """
        Initialize ConnectionManager with configuration.

        Args:
            config: BrainConnectionConfig instance
        """
        self.config = config
        self.brain_id = config.brain_id
        self.connections: Dict[str, Connection] = {}
        # peer_id -> {is_incoming, last_seen, retry_backoff, etc}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Listener for incoming connections
        self.listener: Optional[Listener] = None
        self.listener_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Message queue for received messages (thread-safe)
        self.message_queue: deque = deque()
        self.queue_lock = threading.Lock()

        # Initialize listener if enabled
        if config.enable_listener and config.listen_address:
            self._start_listener()

        # Connect to configured peers
        self._connect_peers()

    def _start_listener(self):
        """Start listening for incoming connections."""
        if self.listener is not None:
            return

        try:
            address = self.config.listen_address
            self.listener = Listener(address, authkey=self.config.authkey)
            print(
                f"[ConnectionManager] Listening on {address} for incoming connections")

            # Start background thread to accept connections
            self.listener_thread = threading.Thread(
                target=self._accept_loop,
                daemon=True,
                name="ConnectionAcceptThread"
            )
            self.listener_thread.start()
        except Exception as e:
            print(f"[ConnectionManager] Failed to start listener: {e}")
            self.listener = None

    def _accept_loop(self):
        """Background thread loop to accept incoming connections."""
        while not self._shutdown.is_set():
            try:
                if self.listener is None:
                    break

                # Try to use select for non-blocking check if possible
                # Fall back to blocking accept with timeout check
                try:
                    # Access underlying socket for non-blocking select
                    listener_fileno = self.listener._listener._socket.fileno()
                    ready, _, _ = select.select([listener_fileno], [], [], 0.5)
                    if not ready:
                        continue  # Check shutdown flag and loop again
                except (AttributeError, OSError):
                    # If we can't access socket, use short sleep before blocking accept
                    # This is less efficient but works
                    if self._shutdown.is_set():
                        break
                    time.sleep(0.1)

                # Accept connection (may block briefly, but we checked shutdown above)
                try:
                    conn = self.listener.accept()
                    self._handle_incoming_connection(conn)
                except OSError:
                    # Socket closed or error
                    break
                except Exception as e:
                    if not self._shutdown.is_set():
                        print(
                            f"[ConnectionManager] Error accepting connection: {e}")
            except Exception as e:
                if not self._shutdown.is_set():
                    print(f"[ConnectionManager] Error in accept loop: {e}")
                time.sleep(0.1)

    def _handle_incoming_connection(self, conn: Connection):
        """Handle a newly accepted incoming connection."""
        # Generate peer_id for incoming connection
        # In future, could use handshake message for peer_id
        peer_id = f"peer_{len(self.connections)}_{int(time.time())}"

        self.connections[peer_id] = conn
        self.connection_metadata[peer_id] = {
            "is_incoming": True,
            "last_seen": time.time(),
            "connected_at": time.time(),
        }

        print(
            f"[ConnectionManager] Accepted incoming connection: {peer_id} from {self.listener.last_accepted if self.listener else 'unknown'}")

    def _connect_peers(self):
        """Connect to all configured peer addresses."""
        for peer_id, host, port in self.config.peers:
            if peer_id not in self.connections:
                self._connect_peer(peer_id, host, port)

    def _connect_peer(self, peer_id: str, host: str, port: int, retry: bool = False):
        """
        Connect to a single peer.

        Args:
            peer_id: Unique identifier for this peer
            host: Host address
            port: Port number
            retry: Whether this is a retry attempt (affects backoff)
        """
        address = (host, port)

        try:
            conn = Client(address, authkey=self.config.authkey)
            self.connections[peer_id] = conn

            metadata = self.connection_metadata.get(peer_id, {})
            metadata.update({
                "is_incoming": False,
                "last_seen": time.time(),
                "connected_at": time.time(),
                "retry_backoff": 1.0,  # Reset backoff on success
                "disconnected": False,
                "address": address,
            })
            self.connection_metadata[peer_id] = metadata

            print(
                f"[ConnectionManager] Connected to peer {peer_id} at {address}")

        except Exception as e:
            if peer_id in self.connections:
                # Connection failed but we had one before
                del self.connections[peer_id]

            metadata = self.connection_metadata.get(peer_id, {})
            if not metadata:
                metadata = {
                    "is_incoming": False,
                    "address": address,
                }

            # Exponential backoff
            backoff = metadata.get("retry_backoff", 1.0)
            if retry:
                backoff = min(backoff * 1.5, 30.0)  # Max 30 seconds
            metadata["retry_backoff"] = backoff
            metadata["disconnected"] = True
            metadata["last_retry"] = time.time()
            metadata["next_retry"] = time.time() + backoff
            self.connection_metadata[peer_id] = metadata

            if not retry:
                print(
                    f"[ConnectionManager] Failed to connect to {peer_id} at {address}: {e}")

    def _reconnect_disconnected(self):
        """Attempt to reconnect to disconnected peers."""
        current_time = time.time()

        for peer_id, metadata in list(self.connection_metadata.items()):
            if metadata.get("disconnected") and not metadata.get("is_incoming"):
                next_retry = metadata.get("next_retry", 0)
                if current_time >= next_retry:
                    address = metadata["address"]
                    self._connect_peer(
                        peer_id, address[0], address[1], retry=True)

    def poll_events(self) -> List[Tuple[str, dict]]:
        """
        Poll for incoming messages from all connections.

        Returns:
            List of (peer_id, message_dict) tuples. Messages include from_id field.
        """
        events = []

        # Check for reconnection opportunities
        self._reconnect_disconnected()

        # Poll all connections for messages
        for peer_id, conn in list(self.connections.items()):
            try:
                if conn.poll(0.0):
                    try:
                        msg = conn.recv()

                        # Add envelope fields
                        if isinstance(msg, dict):
                            msg["from_id"] = peer_id
                            # to_id already set by sender if needed

                            # Update last seen
                            if peer_id in self.connection_metadata:
                                self.connection_metadata[peer_id]["last_seen"] = time.time(
                                )
                                self.connection_metadata[peer_id]["disconnected"] = False

                            events.append((peer_id, msg))
                    except (EOFError, OSError) as e:
                        # Connection lost
                        print(
                            f"[ConnectionManager] Connection to {peer_id} lost: {e}")
                        self._mark_disconnected(peer_id)

            except (EOFError, OSError) as e:
                # Connection error
                print(f"[ConnectionManager] Error polling {peer_id}: {e}")
                self._mark_disconnected(peer_id)

        return events

    def _mark_disconnected(self, peer_id: str):
        """Mark a peer as disconnected and schedule reconnection."""
        if peer_id in self.connections:
            try:
                self.connections[peer_id].close()
            except Exception:
                pass
            del self.connections[peer_id]

        metadata = self.connection_metadata.get(peer_id, {})
        if metadata and not metadata.get("is_incoming"):
            # Only auto-reconnect outgoing connections
            metadata["disconnected"] = True
            backoff = metadata.get("retry_backoff", 1.0)
            metadata["next_retry"] = time.time() + backoff

    def send(self, peer_id: str, message: dict):
        """
        Send a message to a specific peer.

        Args:
            peer_id: Target peer identifier
            message: Message dictionary (will have from_id added automatically)
        """
        if peer_id not in self.connections:
            # Try to reconnect if it's an outgoing peer
            metadata = self.connection_metadata.get(peer_id)
            if metadata and not metadata.get("is_incoming"):
                address = metadata.get("address")
                if address:
                    self._connect_peer(
                        peer_id, address[0], address[1], retry=True)

            if peer_id not in self.connections:
                print(
                    f"[ConnectionManager] Cannot send to {peer_id}: not connected")
                return

        try:
            # Add from_id to message
            msg = message.copy()
            msg["from_id"] = self.brain_id
            # to_id should be set by caller if needed, otherwise we route by peer_id

            conn = self.connections[peer_id]
            conn.send(msg)

            # Update last seen
            if peer_id in self.connection_metadata:
                self.connection_metadata[peer_id]["last_seen"] = time.time()

        except (EOFError, OSError) as e:
            print(f"[ConnectionManager] Failed to send to {peer_id}: {e}")
            self._mark_disconnected(peer_id)
        except Exception as e:
            print(f"[ConnectionManager] Error sending to {peer_id}: {e}")

    def broadcast(self, message: dict):
        """
        Broadcast a message to all connected peers.

        Args:
            message: Message dictionary
        """
        for peer_id in list(self.connections.keys()):
            self.send(peer_id, message)

    def close(self):
        """Close all connections and shutdown listener."""
        self._shutdown.set()

        # Close all connections
        for peer_id, conn in list(self.connections.items()):
            try:
                conn.close()
            except Exception:
                pass

        self.connections.clear()

        # Close listener
        if self.listener:
            try:
                self.listener.close()
            except Exception:
                pass
            self.listener = None

        # Wait for listener thread
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)

        print("[ConnectionManager] All connections closed")
