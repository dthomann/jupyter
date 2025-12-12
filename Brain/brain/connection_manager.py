"""
ConnectionManager for bi-directional brain communication using UDP multicast discovery.

Manages both incoming (Listener) and outgoing (Client) connections,
providing a unified interface for message passing.

Each node (brain or environment) is fully symmetric - listening and talking.
- Uses UDP multicast for peer discovery (HELLO messages)
- Connects to discovered peers via TCP
- Application messages only sent over TCP to connected peers
"""

import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from multiprocessing.connection import Listener, Client, Connection
import select
import socket

from .connection_config import BrainConnectionConfig
from .multicast_discovery import MulticastDiscovery, PeerInfo


class ConnectionManager:
    """
    Manages all connections for a brain or environment process using multicast discovery.

    Supports:
    - UDP multicast discovery (HELLO messages)
    - TCP listener for incoming connections
    - TCP client connections for outgoing connections
    - Non-blocking message polling
    - Automatic connection to discovered peers
    - Message envelope handling (from_id, to_id)
    """

    def __init__(
        self,
        config: BrainConnectionConfig,
        peer_type: str = "brain",
        multicast_group: str = "239.0.0.1",
        multicast_port: int = 50000,
        max_brains: Optional[int] = None,
    ):
        """
        Initialize ConnectionManager with configuration.

        Args:
            config: BrainConnectionConfig instance
            peer_type: Type of this peer ("brain" or "environment")
            multicast_group: Multicast group address for discovery
            multicast_port: Multicast UDP port for discovery
            max_brains: Maximum number of brain connections (for environments only)
        """
        self.config = config
        self.node_id = config.brain_id
        self.peer_type = peer_type
        self.max_brains = max_brains
        self.connections: Dict[str, Connection] = {}
        # peer_id -> {is_incoming, last_seen, retry_backoff, peer_type, address, etc}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Listener for incoming connections
        self.listener: Optional[Listener] = None
        self.listener_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Message queue for received messages (thread-safe)
        self.message_queue: deque = deque()
        self.queue_lock = threading.Lock()

        # Lock for thread-safe connection management
        self.connection_lock = threading.Lock()

        # Get listen address
        if not config.listen_address:
            # Auto-assign if not specified
            import socket as sock
            with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', 0))
                _, port = s.getsockname()
                config.listen_address = ('0.0.0.0', port)

        listen_host, listen_port = config.listen_address

        # Initialize multicast discovery
        self.discovery = MulticastDiscovery(
            node_id=self.node_id,
            kind=peer_type,
            listen_host=listen_host,
            listen_port=listen_port,
            multicast_group=multicast_group,
            multicast_port=multicast_port,
            on_peer_discovered=self._on_peer_discovered,
            on_peer_expired=self._on_peer_expired,
        )

        # Initialize listener
        self._start_listener()

        # Start discovery
        self.discovery.start()

    def _start_listener(self):
        """Start listening for incoming connections."""
        if self.listener is not None:
            return

        try:
            address = self.config.listen_address
            self.listener = Listener(address, authkey=self.config.authkey)
            print(
                f"[ConnectionManager] {self.peer_type.capitalize()} {self.node_id} listening on {address}")

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

                try:
                    listener_fileno = self.listener._listener._socket.fileno()
                    ready, _, _ = select.select([listener_fileno], [], [], 0.5)
                    if not ready:
                        continue
                except (AttributeError, OSError):
                    if self._shutdown.is_set():
                        break
                    time.sleep(0.1)

                try:
                    conn = self.listener.accept()
                    self._handle_incoming_connection(conn)
                except OSError:
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
        # Note: We don't check the brain limit here because we don't know if this
        # connection is from a brain yet. The limit is checked in poll_events()
        # after we receive the startup message and identify the peer type.

        with self.connection_lock:
            # Generate temporary peer_id - will be updated when we receive peer info
            # Use a more unique ID to avoid collisions (microsecond precision + connection count)
            base_id = f"incoming_{len(self.connections)}_{int(time.time() * 1000000)}"
            peer_id = base_id
            counter = 0
            # Ensure uniqueness (shouldn't happen, but be safe)
            while peer_id in self.connections:
                counter += 1
                peer_id = f"{base_id}_{counter}"

            self.connections[peer_id] = conn
            self.connection_metadata[peer_id] = {
                "is_incoming": True,
                "last_seen": time.time(),
                "connected_at": time.time(),
                "temp_id": True,
                "startup_sent": False,
            }

        print(
            f"[ConnectionManager] Accepted incoming connection: {peer_id}")

    def _update_peer_id(self, old_peer_id: str, new_peer_id: str, peer_info: Dict[str, Any]):
        """Update peer_id mapping when we receive peer information."""
        if old_peer_id == new_peer_id:
            return

        with self.connection_lock:
            # Check if new_peer_id already has a connection (duplicate connection scenario)
            if new_peer_id in self.connections:
                old_conn = self.connections[new_peer_id]
                print(
                    f"[ConnectionManager] WARNING: peer_id {new_peer_id} already has a connection. Closing old connection.")
                try:
                    old_conn.close()
                except Exception:
                    pass
                # Remove old connection metadata if it exists
                if new_peer_id in self.connection_metadata:
                    del self.connection_metadata[new_peer_id]

            # Move connection from old_peer_id to new_peer_id
            if old_peer_id in self.connections:
                self.connections[new_peer_id] = self.connections.pop(
                    old_peer_id)

            # Move metadata
            if old_peer_id in self.connection_metadata:
                metadata = self.connection_metadata.pop(old_peer_id)
                metadata.update(peer_info)
                metadata["temp_id"] = False
                self.connection_metadata[new_peer_id] = metadata
                print(
                    f"[ConnectionManager] Updated peer {old_peer_id} -> {new_peer_id}, peer_type={peer_info.get('peer_type')}")

        # Update discovery peer connection
        self.discovery.set_peer_connection(
            new_peer_id, self.connections.get(new_peer_id))

    def _on_peer_discovered(self, node_id: str, kind: str, host: str, port: int, same_kind_count: int):
        """Callback when a new peer is discovered via multicast."""
        print(
            f"[ConnectionManager] {self.peer_type.capitalize()} {self.node_id} discovered {kind} {node_id} at {host}:{port}")

        # Check if we should connect
        should_connect = False

        if self.peer_type == "brain":
            # Brain only connects to environments, and only if exactly one exists
            if kind == "environment":
                # Use the count passed from discovery (avoids deadlock)
                env_count = same_kind_count
                print(
                    f"[ConnectionManager] Brain {self.node_id} checking environment count: {env_count}")
                if env_count == 1:
                    should_connect = True
                    print(
                        f"[ConnectionManager] Detected single environment {node_id}, connecting...")
                else:
                    print(
                        f"[ConnectionManager] {env_count} environments detected, not auto-connecting")
            else:
                print(
                    f"[ConnectionManager] Brain discovered {kind} {node_id}, not connecting (only connect to environments)")
            # Brains don't auto-connect to other brains
        elif self.peer_type == "environment":
            # Environments do NOT proactively connect to brains
            # They only accept incoming connections from brains that connect to them
            if kind == "brain":
                print(
                    f"[ConnectionManager] Environment discovered brain {node_id}, waiting for brain to connect...")
                should_connect = False
            else:
                print(
                    f"[ConnectionManager] Environment discovered {kind} {node_id}, not connecting")

        if should_connect:
            if node_id in self.connections:
                print(
                    f"[ConnectionManager] Already connected to {node_id}, skipping")
            else:
                print(
                    f"[ConnectionManager] Starting connection thread to {node_id}...")
                # Attempt connection in background
                threading.Thread(
                    target=self._connect_to_peer,
                    args=(node_id, host, port, kind),
                    daemon=True,
                    name=f"ConnectTo{node_id}"
                ).start()
        else:
            print(
                f"[ConnectionManager] Not connecting to {node_id} (should_connect=False)")

    def _on_peer_expired(self, node_id: str):
        """Callback when a peer expires (no HELLO received)."""
        # Mark as disconnected
        self._mark_disconnected(node_id)

    def _connect_to_peer(self, node_id: str, host: str, port: int, kind: str):
        """Attempt to connect to a peer."""
        address = (host, port)

        try:
            conn = Client(address, authkey=self.config.authkey)

            # Thread-safe check and assignment
            with self.connection_lock:
                # Double-check: another thread might have connected while we were establishing connection
                if node_id in self.connections:
                    print(
                        f"[ConnectionManager] Already connected to {node_id}, closing duplicate connection attempt")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    return  # Already connected, exit

                # Assign connection
                self.connections[node_id] = conn

                metadata = {
                    "is_incoming": False,
                    "last_seen": time.time(),
                    "connected_at": time.time(),
                    "retry_backoff": 1.0,
                    "disconnected": False,
                    "address": address,
                    "peer_type": kind,
                    "temp_id": False,
                    "startup_sent": False,
                }
                self.connection_metadata[node_id] = metadata

            # Update discovery (outside lock to avoid deadlock)
            self.discovery.set_peer_connection(node_id, conn)

            print(
                f"[ConnectionManager] {self.peer_type.capitalize()} {self.node_id} connected to {kind} {node_id} at {address} (peer_type set to: {kind})")

            # Send startup message over TCP
            self._send_startup_message(node_id)

        except Exception as e:
            print(
                f"[ConnectionManager] Failed to connect to {node_id} at {address}: {e}")
            with self.connection_lock:
                if node_id in self.connections:
                    del self.connections[node_id]
                if node_id in self.connection_metadata:
                    del self.connection_metadata[node_id]
            # Will retry on next HELLO if peer is still alive

    def _send_startup_message(self, peer_id: str):
        """Send startup message to a peer over TCP (only once per peer)."""
        # Check if we've already sent a startup message to this peer
        if peer_id in self.connection_metadata:
            if self.connection_metadata[peer_id].get("startup_sent", False):
                return  # Already sent, don't send again

        listen_addr = self.get_listen_address()
        if listen_addr and peer_id in self.connections:
            startup_msg = {
                "type": "discovery/startup",
                "peer_id": self.node_id,
                "peer_type": self.peer_type,
                "listen_address": listen_addr,
            }
            try:
                # Send directly without adding from_id (send() will add it)
                msg = startup_msg.copy()
                msg["from_id"] = self.node_id
                msg["peer_id"] = self.node_id
                msg["peer_type"] = self.peer_type
                self.connections[peer_id].send(msg)

                # Mark as sent
                if peer_id not in self.connection_metadata:
                    self.connection_metadata[peer_id] = {}
                self.connection_metadata[peer_id]["startup_sent"] = True

                print(f"[ConnectionManager] Sent startup message to {peer_id}")
            except Exception as e:
                print(
                    f"[ConnectionManager] Failed to send startup to {peer_id}: {e}")

    def poll_events(self) -> List[Tuple[str, dict]]:
        """
        Poll for incoming messages from all connections.

        Returns:
            List of (peer_id, message_dict) tuples. Messages include from_id field.
        """
        events = []

        # Get snapshot of connections (with lock) to avoid modification during iteration
        with self.connection_lock:
            connections_snapshot = list(self.connections.items())

        # Poll all connections for messages
        for peer_id, conn in connections_snapshot:
            try:
                if conn.poll(0.0):
                    try:
                        msg = conn.recv()

                        if isinstance(msg, dict):
                            # Extract sender info from message
                            sender_peer_id = msg.get(
                                "peer_id") or msg.get("from_id")
                            sender_peer_type = msg.get("peer_type", "unknown")

                            # Message logging disabled
                            # msg_type = msg.get("type", "unknown")
                            # if not msg_type.startswith("discovery/"):
                            #     # Log the message with sender and receiver info
                            #     sender_info = f"{sender_peer_type} {sender_peer_id or peer_id}"
                            #     receiver_info = f"{self.peer_type} {self.node_id}"
                            #     print(
                            #         f"[MSG_RECV] {sender_info} -> {receiver_info}: type={msg_type}")
                            #
                            #     # Log message content (truncate large arrays)
                            #     if msg_type == "observation":
                            #         sensors = msg.get("sensors", [])
                            #         if isinstance(sensors, list) and len(sensors) > 10:
                            #             print(
                            #                 f"  sensors=[{sensors[:5]}... ({len(sensors)} total)]")
                            #         else:
                            #             print(f"  sensors={sensors}")
                            #         print(f"  info={msg.get('info', {})}")
                            #     elif msg_type == "action":
                            #         actions = msg.get("actions", [])
                            #         print(f"  actions={actions}")
                            #         print(f"  info={msg.get('info', {})}")
                            #     elif msg_type == "reward":
                            #         print(f"  value={msg.get('value')}")
                            #         print(f"  info={msg.get('info', {})}")
                            #     elif msg_type == "terminal":
                            #         print(f"  info={msg.get('info', {})}")
                            #     else:
                            #         # For other message types, log the full message (but limit size)
                            #         msg_str = str(msg)
                            #         if len(msg_str) > 200:
                            #             print(f"  {msg_str[:200]}...")
                            #         else:
                            #             print(f"  {msg_str}")

                            # Update peer_id if this was a temporary incoming connection
                            if peer_id.startswith("incoming_") and sender_peer_id and sender_peer_id != peer_id:
                                # For environments, check brain connection limit before accepting new brain
                                if (self.peer_type == "environment" and
                                    sender_peer_type == "brain" and
                                        self.max_brains is not None):
                                    # Count existing brain connections (excluding this temporary one)
                                    brain_count = sum(
                                        1 for pid, metadata in self.connection_metadata.items()
                                        if metadata.get("peer_type") == "brain" and pid != peer_id
                                    )
                                    if brain_count >= self.max_brains:
                                        print(
                                            f"[ConnectionManager] Environment {self.node_id} reached max_brains limit ({self.max_brains}), rejecting brain {sender_peer_id}")
                                        # Disconnect using temp peer_id
                                        self._mark_disconnected(peer_id)
                                        continue  # Skip processing this message

                                print(
                                    f"[ConnectionManager] Updating peer_id from {peer_id} to {sender_peer_id} (type: {sender_peer_type})")
                                self._update_peer_id(peer_id, sender_peer_id, {
                                    "peer_type": sender_peer_type,
                                    "address": msg.get("listen_address"),
                                })
                                peer_id = sender_peer_id

                            # Handle startup message - reply with our presence (only if we haven't already)
                            if msg.get("type") == "discovery/startup" and sender_peer_id:
                                print(
                                    f"[ConnectionManager] Received startup message from {peer_id} (type: {sender_peer_type})")
                                # Reply with our startup message (only once per peer)
                                self._send_startup_message(peer_id)

                            msg["from_id"] = peer_id

                            # Update last seen (with lock)
                            with self.connection_lock:
                                if peer_id in self.connection_metadata:
                                    self.connection_metadata[peer_id]["last_seen"] = time.time(
                                    )
                                    self.connection_metadata[peer_id]["disconnected"] = False

                            events.append((peer_id, msg))
                    except (EOFError, OSError) as e:
                        print(
                            f"[ConnectionManager] Connection to {peer_id} lost: {e}")
                        self._mark_disconnected(peer_id)

            except (EOFError, OSError) as e:
                print(f"[ConnectionManager] Error polling {peer_id}: {e}")
                self._mark_disconnected(peer_id)

        return events

    def _mark_disconnected(self, peer_id: str):
        """Mark a peer as disconnected."""
        with self.connection_lock:
            if peer_id in self.connections:
                try:
                    self.connections[peer_id].close()
                except Exception:
                    pass
                del self.connections[peer_id]

            if peer_id in self.connection_metadata:
                self.connection_metadata[peer_id]["disconnected"] = True

        # Update discovery (outside lock to avoid deadlock)
        self.discovery.set_peer_connection(peer_id, None)

    def send(self, peer_id: str, message: dict):
        """
        Send a message to a specific peer over TCP.

        Args:
            peer_id: Target peer identifier
            message: Message dictionary (will have from_id added automatically)
        """
        # Get connection (with lock to ensure it still exists)
        with self.connection_lock:
            if peer_id not in self.connections:
                print(
                    f"[ConnectionManager] Cannot send to {peer_id}: not connected")
                return
            conn = self.connections[peer_id]

        try:
            msg = message.copy()
            msg["from_id"] = self.node_id
            msg["peer_id"] = self.node_id
            msg["peer_type"] = self.peer_type

            # Message logging disabled
            # msg_type = msg.get("type", "unknown")
            # if not msg_type.startswith("discovery/"):
            #     # Log the message with sender and receiver info
            #     sender_info = f"{self.peer_type} {self.node_id}"
            #     receiver_info = f"{peer_id}"
            #     print(
            #         f"[MSG_SEND] {sender_info} -> {receiver_info}: type={msg_type}")
            #
            #     # Log message content (truncate large arrays)
            #     if msg_type == "observation":
            #         sensors = msg.get("sensors", [])
            #         if isinstance(sensors, list) and len(sensors) > 10:
            #             print(
            #                 f"  sensors=[{sensors[:5]}... ({len(sensors)} total)]")
            #         else:
            #             print(f"  sensors={sensors}")
            #         print(f"  info={msg.get('info', {})}")
            #     elif msg_type == "action":
            #         actions = msg.get("actions", [])
            #         print(f"  actions={actions}")
            #         print(f"  info={msg.get('info', {})}")
            #     elif msg_type == "reward":
            #         print(f"  value={msg.get('value')}")
            #         print(f"  info={msg.get('info', {})}")
            #     elif msg_type == "terminal":
            #         print(f"  info={msg.get('info', {})}")
            #     else:
            #         # For other message types, log the full message (but limit size)
            #         msg_str = str(msg)
            #         if len(msg_str) > 200:
            #             print(f"  {msg_str[:200]}...")
            #         else:
            #             print(f"  {msg_str}")

            conn.send(msg)

            # Update last_seen (with lock)
            with self.connection_lock:
                if peer_id in self.connection_metadata:
                    self.connection_metadata[peer_id]["last_seen"] = time.time(
                    )

        except (EOFError, OSError) as e:
            print(f"[ConnectionManager] Failed to send to {peer_id}: {e}")
            self._mark_disconnected(peer_id)
        except Exception as e:
            print(f"[ConnectionManager] Error sending to {peer_id}: {e}")

    def broadcast(self, message: dict):
        """
        Broadcast a message to all connected peers over TCP.

        Args:
            message: Message dictionary
        """
        # Get snapshot of peer_ids (with lock)
        with self.connection_lock:
            peer_ids = list(self.connections.keys())

        for peer_id in peer_ids:
            self.send(peer_id, message)

    def list_available_peers(self) -> List[Dict[str, Any]]:
        """
        Get list of currently connected peers with their metadata.

        Returns:
            List of dicts with peer_id, peer_type, is_incoming, address, last_seen, etc.
        """
        peers = []
        with self.connection_lock:
            for peer_id, metadata in self.connection_metadata.items():
                if peer_id in self.connections:
                    peer_info = {
                        "peer_id": peer_id,
                        "peer_type": metadata.get("peer_type", "unknown"),
                        "is_incoming": metadata.get("is_incoming", False),
                        "last_seen": metadata.get("last_seen", 0),
                        "connected_at": metadata.get("connected_at", 0),
                    }
                    if "address" in metadata:
                        peer_info["address"] = metadata["address"]
                    peers.append(peer_info)
        return peers

    def get_listen_address(self) -> Optional[Tuple[str, int]]:
        """Get the address this node is listening on."""
        if self.listener and self.config.listen_address:
            return self.config.listen_address
        return None

    def get_peer_type(self, peer_id: str) -> Optional[str]:
        """Get the type of a peer."""
        with self.connection_lock:
            return self.connection_metadata.get(peer_id, {}).get("peer_type")

    def get_all_known_peers(self) -> Dict[str, PeerInfo]:
        """Get all known peers from discovery (including not connected)."""
        return self.discovery.get_peers()

    def close(self):
        """Close all connections and shutdown listener."""
        # Send shutdown messages to all known peers (attempt to connect if needed)
        known_peers = self.discovery.get_peers()
        shutdown_msg = {
            "type": "discovery/shutdown",
            "peer_id": self.node_id,
            "peer_type": self.peer_type,
        }

        # Send to connected peers
        with self.connection_lock:
            peer_ids = list(self.connections.keys())

        for peer_id in peer_ids:
            try:
                self.send(peer_id, shutdown_msg)
            except Exception:
                pass

        # Attempt to connect and send to known but not connected peers
        with self.connection_lock:
            connected_peer_ids = set(self.connections.keys())

        for node_id, peer_info in known_peers.items():
            if node_id not in connected_peer_ids and node_id != self.node_id:
                try:
                    conn = Client((peer_info.host, peer_info.port),
                                  authkey=self.config.authkey)
                    msg = shutdown_msg.copy()
                    msg["from_id"] = self.node_id
                    conn.send(msg)
                    conn.close()
                except Exception:
                    pass  # Ignore failures on shutdown

        self._shutdown.set()

        # Stop discovery
        self.discovery.stop()

        # Close all connections
        with self.connection_lock:
            connections_to_close = list(self.connections.items())
            self.connections.clear()
            self.connection_metadata.clear()

        for peer_id, conn in connections_to_close:
            try:
                conn.close()
            except Exception:
                pass

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

        print(
            f"[ConnectionManager] {self.peer_type.capitalize()} {self.node_id} closed all connections")
