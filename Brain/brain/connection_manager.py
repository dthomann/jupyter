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

        # Track outbound connection attempts to avoid spawning many duplicate threads
        self._connect_in_progress: set[str] = set()
        self._connect_state_lock = threading.Lock()

        # Rate-limit discovery logging (HELLO callbacks can be frequent)
        self._last_discovery_log: Dict[str, float] = {}
        self._last_env_count: Optional[int] = None
        self._last_env_decision_log: float = 0.0

        # Get listen address
        if not config.listen_address:
            # Auto-assign if not specified
            import socket as sock
            with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', 0))
                _, port = s.getsockname()
                config.listen_address = ('0.0.0.0', port)

        listen_host, listen_port = config.listen_address

        # Choose a host to advertise in multicast HELLO.
        # If we're bound to 0.0.0.0, that is not a routable destination address.
        advertise_host = self._choose_advertise_host(
            listen_host, multicast_group, multicast_port)

        # Initialize multicast discovery
        self.discovery = MulticastDiscovery(
            node_id=self.node_id,
            kind=peer_type,
            listen_host=advertise_host,
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

    @staticmethod
    def _choose_advertise_host(listen_host: str, multicast_group: str, multicast_port: int) -> str:
        """
        Pick a reasonable host/IP to advertise to peers.

        Binding to 0.0.0.0 means "all interfaces" and cannot be used as a destination.
        We try to infer the primary local IP without sending any packets.
        """
        if listen_host and listen_host not in ("0.0.0.0", "::"):
            return listen_host

        # Try to infer the local IP for the route we'd use to reach the multicast group.
        for target in ((multicast_group, multicast_port), ("1.1.1.1", 80), ("8.8.8.8", 80)):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    s.connect(target)
                    ip = s.getsockname()[0]
                    if ip and ip != "0.0.0.0":
                        return ip
                finally:
                    s.close()
            except Exception:
                continue

        # Safe fallback for single-host setups.
        return "127.0.0.1"

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
            # If new_peer_id already exists, fully clean it up first (this handles reconnections)
            # When a brain reconnects, we want to treat it as a completely new connection
            if new_peer_id in self.connections:
                old_conn = self.connections[new_peer_id]
                print(
                    f"[ConnectionManager] peer_id {new_peer_id} already exists (likely reconnection). Fully cleaning up old connection.")
                try:
                    old_conn.close()
                except Exception:
                    pass
                del self.connections[new_peer_id]

            # Also clean up any existing metadata for this peer_id
            if new_peer_id in self.connection_metadata:
                del self.connection_metadata[new_peer_id]

            # Move connection from old_peer_id to new_peer_id
            if old_peer_id in self.connections:
                self.connections[new_peer_id] = self.connections.pop(
                    old_peer_id)
            else:
                # This shouldn't happen, but if old_peer_id is not in connections,
                # the connection might have been cleaned up. Log a warning.
                print(
                    f"[ConnectionManager] WARNING: old_peer_id {old_peer_id} not in connections when updating to {new_peer_id}. Connection may have been cleaned up.")

            # Move or create metadata
            if old_peer_id in self.connection_metadata:
                metadata = self.connection_metadata.pop(old_peer_id)
                metadata.update(peer_info)
            else:
                # Create new metadata if it doesn't exist
                metadata = peer_info.copy()
            metadata["temp_id"] = False
            self.connection_metadata[new_peer_id] = metadata
            print(
                f"[ConnectionManager] Updated peer {old_peer_id} -> {new_peer_id}, peer_type={peer_info.get('peer_type')}")

        # Update discovery peer connection
        try:
            conn = self.connections.get(new_peer_id)
            # Check if peer exists in discovery (with lock to avoid race conditions)
            # Use non-blocking lock acquisition to avoid deadlocks
            peer_exists = False
            try:
                # Try to acquire lock without blocking
                lock_acquired = self.discovery.peers_lock.acquire(
                    blocking=False)
                if lock_acquired:
                    try:
                        peer_exists = new_peer_id in self.discovery.peers
                    finally:
                        self.discovery.peers_lock.release()
                else:
                    # Lock is held by another thread - skip the check and assume peer doesn't exist
                    # This avoids blocking and potential deadlocks
                    peer_exists = False
            except Exception:
                # If lock acquisition fails for any reason, assume peer doesn't exist to avoid blocking
                peer_exists = False
            # Only call set_peer_connection if the peer exists in discovery
            # This avoids blocking on the lock when the peer hasn't been discovered yet
            if peer_exists:
                self.discovery.set_peer_connection(new_peer_id, conn)
        except Exception as e:
            # Never raise from background/IO threads; just log and continue.
            print(
                f"[ConnectionManager] WARNING: failed to update discovery connection for {new_peer_id}: {e}")

    def _on_peer_discovered(self, node_id: str, kind: str, host: str, port: int, same_kind_count: int):
        """Callback when a new peer is discovered via multicast."""
        # We may get called on every HELLO. Keep this callback side-effect free unless
        # we actually need to (re)connect, otherwise logs become misleading.
        now = time.time()

        # Check if we should connect
        should_connect = False

        if self.peer_type == "brain":
            # Brain only connects to environments, and only if exactly one exists
            if kind == "environment":
                # Use the count passed from discovery (avoids deadlock)
                env_count = same_kind_count
                self._last_env_count = env_count
                if env_count == 1:
                    should_connect = True
                else:
                    # Only log this occasionally, otherwise it spams every HELLO.
                    if now - self._last_env_decision_log > 5.0:
                        print(
                            f"[ConnectionManager] Brain {self.node_id}: {env_count} environments detected; waiting for exactly 1 before auto-connect")
                        self._last_env_decision_log = now
            else:
                # Brains don't connect to other kinds here.
                pass
            # Brains don't auto-connect to other brains
        elif self.peer_type == "environment":
            # Environments do NOT proactively connect to brains
            # They only accept incoming connections from brains that connect to them
            should_connect = False

        if should_connect:
            # Only attempt when not already connected; _maybe_connect_to_peer also
            # applies dedupe + backoff.
            with self.connection_lock:
                already_connected = node_id in self.connections
            if not already_connected:
                # Rate-limit "connect attempt" logs per peer
                last = self._last_discovery_log.get(node_id, 0.0)
                if now - last > 3.0:
                    print(
                        f"[ConnectionManager] {self.peer_type.capitalize()} {self.node_id} attempting connect to {kind} {node_id} at {host}:{port}")
                    self._last_discovery_log[node_id] = now
            self._maybe_connect_to_peer(node_id, host, port, kind)

    def _on_peer_expired(self, node_id: str):
        """Callback when a peer expires (no HELLO received)."""
        # Mark as disconnected
        self._mark_disconnected(node_id, keep_metadata=False)

    def _maybe_connect_to_peer(self, node_id: str, host: str, port: int, kind: str):
        """Schedule an outbound connect attempt with dedupe + backoff."""
        address = (host, port)
        now = time.time()

        with self.connection_lock:
            if node_id in self.connections:
                return

            md = self.connection_metadata.get(node_id, {})
            # Backoff state
            next_retry_at = float(md.get("next_retry_at", 0.0) or 0.0)
            if now < next_retry_at:
                return

        with self._connect_state_lock:
            if node_id in self._connect_in_progress:
                return
            self._connect_in_progress.add(node_id)

        # Attempt connection in background
        threading.Thread(
            target=self._connect_to_peer,
            args=(node_id, host, port, kind),
            daemon=True,
            name=f"ConnectTo{node_id}"
        ).start()

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
            # Record backoff so we don't spin aggressively.
            now = time.time()
            with self.connection_lock:
                if node_id in self.connections:
                    try:
                        self.connections[node_id].close()
                    except Exception:
                        pass
                    del self.connections[node_id]

                md = self.connection_metadata.get(node_id, {})
                backoff = float(md.get("retry_backoff", 1.0) or 1.0)
                # Exponential backoff with cap
                backoff = min(backoff * 2.0, 10.0)
                md.update({
                    "is_incoming": False,
                    "disconnected": True,
                    "address": address,
                    "peer_type": kind,
                    "retry_backoff": backoff,
                    "last_failed_connect_at": now,
                    "next_retry_at": now + backoff,
                })
                self.connection_metadata[node_id] = md
            # Retry will be scheduled by subsequent HELLOs.
        finally:
            with self._connect_state_lock:
                self._connect_in_progress.discard(node_id)

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
        # Batch last_seen updates to reduce lock contention
        last_seen_updates = {}

        for peer_id, conn in connections_snapshot:
            try:
                if conn.poll(0.0):
                    try:
                        msg = conn.recv()
                        try:

                            if isinstance(msg, dict):
                                # Extract sender info from message
                                sender_peer_id = msg.get(
                                    "peer_id") or msg.get("from_id")
                                sender_peer_type = msg.get(
                                    "peer_type", "unknown")

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
                                # Only update on startup messages, not shutdown messages (shutdown messages are from old connections)
                                peer_id_was_updated = False
                                if (peer_id.startswith("incoming_") and sender_peer_id and sender_peer_id != peer_id and
                                        msg.get("type") == "discovery/startup"):
                                    # For environments, check brain connection limit before accepting new brain
                                    if (self.peer_type == "environment" and
                                        sender_peer_type == "brain" and
                                            self.max_brains is not None):
                                        # Count existing brain connections (excluding this temporary one)
                                        # Only count brains that are actually connected (in self.connections)
                                        with self.connection_lock:
                                            brain_count = sum(
                                                1 for pid, metadata in self.connection_metadata.items()
                                                if (pid != peer_id and
                                                    pid in self.connections and
                                                    metadata.get("peer_type") == "brain")
                                            )
                                        if brain_count >= self.max_brains:
                                            print(
                                                f"[ConnectionManager] Environment {self.node_id} reached max_brains limit ({self.max_brains}), rejecting brain {sender_peer_id}")
                                            # Disconnect using temp peer_id
                                            self._mark_disconnected(peer_id)
                                            continue  # Skip processing this message

                                    print(
                                        f"[ConnectionManager] Updating peer_id from {peer_id} to {sender_peer_id} (type: {sender_peer_type})")
                                    try:
                                        self._update_peer_id(peer_id, sender_peer_id, {
                                            "peer_type": sender_peer_type,
                                            "address": msg.get("listen_address"),
                                        })
                                    except Exception as e:
                                        raise
                                    peer_id = sender_peer_id
                                    peer_id_was_updated = True

                                # Handle startup message - reply with our presence (only if we haven't already)
                                # Only process actual startup messages as startup. A reconnecting brain sends a startup message,
                                # not a shutdown message. Shutdown messages are from old connections and should be ignored.
                                # Only process actual startup messages as startup
                                if msg.get("type") == "discovery/startup" and sender_peer_id:
                                    print(
                                        f"[ConnectionManager] Received startup message from {peer_id} (type: {sender_peer_type})")
                                    # Reply with our startup message (only once per peer)
                                    self._send_startup_message(peer_id)

                                msg["from_id"] = peer_id
                                # Batch last_seen update
                                last_seen_updates[peer_id] = time.time()
                                events.append((peer_id, msg))
                        except Exception as e:
                            print(
                                f"[ConnectionManager] Error processing message from {peer_id}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    except (EOFError, OSError) as e:
                        print(
                            f"[ConnectionManager] Connection to {peer_id} lost: {e}")
                        self._mark_disconnected(
                            peer_id, keep_metadata=not peer_id.startswith("incoming_"))

            except (EOFError, OSError) as e:
                print(f"[ConnectionManager] Error polling {peer_id}: {e}")
                self._mark_disconnected(
                    peer_id, keep_metadata=not peer_id.startswith("incoming_"))

        # Batch update last_seen for all processed messages
        if last_seen_updates:
            with self.connection_lock:
                for pid, timestamp in last_seen_updates.items():
                    if pid in self.connection_metadata:
                        self.connection_metadata[pid]["last_seen"] = timestamp
                        self.connection_metadata[pid]["disconnected"] = False

        return events

    def _mark_disconnected(self, peer_id: str, keep_metadata: bool = True):
        """Mark a peer as disconnected."""
        with self.connection_lock:
            if peer_id in self.connections:
                try:
                    self.connections[peer_id].close()
                except Exception:
                    pass
                del self.connections[peer_id]

            if peer_id in self.connection_metadata:
                if keep_metadata:
                    self.connection_metadata[peer_id]["disconnected"] = True
                    self.connection_metadata[peer_id]["last_disconnected"] = time.time(
                    )
                else:
                    # Remove metadata for expired peers / temp connections
                    del self.connection_metadata[peer_id]

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
            self._mark_disconnected(
                peer_id, keep_metadata=not peer_id.startswith("incoming_"))
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
