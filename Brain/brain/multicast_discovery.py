"""
UDP Multicast Discovery for peer-to-peer node discovery.

Each node (brain or environment) joins a multicast group and:
- Periodically broadcasts HELLO messages
- Listens for HELLO messages from other nodes
- Maintains a peer table of discovered nodes
"""

import socket
import json
import threading
import time
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class PeerInfo:
    """Information about a discovered peer."""
    node_id: str
    kind: str  # "brain" or "env"
    host: str
    port: int
    last_seen: float
    tcp_conn: Optional[object] = None  # TCP connection object if connected


class MulticastDiscovery:
    """
    Handles UDP multicast discovery for peer-to-peer networking.

    Each node broadcasts HELLO messages and listens for HELLOs from others.
    """

    def __init__(
        self,
        node_id: str,
        kind: str,  # "brain" or "env"
        listen_host: str,
        listen_port: int,
        multicast_group: str = "239.0.0.1",
        multicast_port: int = 50000,
        hello_interval: float = 1.0,
        peer_timeout: float = 5.0,
        on_peer_discovered: Optional[Callable[[
            str, str, str, int, int], None]] = None,  # node_id, kind, host, port, same_kind_count
        on_peer_expired: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize multicast discovery.

        Args:
            node_id: Unique identifier for this node
            kind: Type of node ("brain" or "env")
            listen_host: Host address where this node's TCP listener is running
            listen_port: Port where this node's TCP listener is running
            multicast_group: Multicast group address (default: 239.0.0.1)
            multicast_port: Multicast UDP port (default: 50000)
            hello_interval: Seconds between HELLO broadcasts (default: 1.0)
            peer_timeout: Seconds before considering a peer expired (default: 5.0)
            on_peer_discovered: Callback when a new peer is discovered (node_id, kind, host, port)
            on_peer_expired: Callback when a peer expires (node_id)
        """
        self.node_id = node_id
        self.kind = kind
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self.hello_interval = hello_interval
        self.peer_timeout = peer_timeout

        self.on_peer_discovered = on_peer_discovered
        self.on_peer_expired = on_peer_expired

        # Peer table: node_id -> PeerInfo
        self.peers: Dict[str, PeerInfo] = {}
        self.peers_lock = threading.Lock()

        # Control flags
        self._running = False
        self._shutdown = threading.Event()

        # Threads
        self._sender_thread: Optional[threading.Thread] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._expirer_thread: Optional[threading.Thread] = None

        # UDP socket
        self._sock: Optional[socket.socket] = None

    def start(self):
        """Start the discovery service."""
        if self._running:
            return

        try:
            # Create UDP socket
            self._sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

            # Enable address reuse (required for multiple processes to bind to same multicast port)
            # Must be set BEFORE binding
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # On macOS and Linux, also need SO_REUSEPORT for multiple processes to bind to same port
            # SO_REUSEPORT might not be in socket module, so use the constant value directly
            # Value is 0x0200 on most systems (macOS, Linux, BSD)
            try:
                SO_REUSEPORT = getattr(socket, 'SO_REUSEPORT', 0x0200)
                self._sock.setsockopt(socket.SOL_SOCKET, SO_REUSEPORT, 1)
            except (OSError, AttributeError) as e:
                # If SO_REUSEPORT fails, try to continue anyway
                # On some systems it might not be needed
                pass

            # For UDP multicast, bind to INADDR_ANY (0.0.0.0) and the multicast port
            # Multiple processes can bind to the same port with SO_REUSEADDR
            # The multicast group membership will filter which packets we receive
            self._sock.bind(('0.0.0.0', self.multicast_port))

            # Join multicast group - this tells the OS to deliver multicast packets to this socket
            # Use INADDR_ANY to receive from any interface
            group = socket.inet_aton(self.multicast_group)
            mreq = group + socket.inet_aton('0.0.0.0')
            self._sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

            # Set multicast TTL (time to live) - allow packets to traverse networks if needed
            # TTL=1 means packets stay on local network
            self._sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

            # Set multicast loopback - allow receiving our own packets (useful for testing)
            self._sock.setsockopt(
                socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

            # Set socket to non-blocking for timeout-based operations
            self._sock.settimeout(1.0)

            self._running = True
            self._shutdown.clear()

            # Start threads
            self._sender_thread = threading.Thread(
                target=self._sender_loop, daemon=True, name="MulticastSender")
            self._listener_thread = threading.Thread(
                target=self._listener_loop, daemon=True, name="MulticastListener")
            self._expirer_thread = threading.Thread(
                target=self._expirer_loop, daemon=True, name="PeerExpirer")

            self._sender_thread.start()
            self._listener_thread.start()
            self._expirer_thread.start()

            print(
                f"[MulticastDiscovery] {self.kind.capitalize()} {self.node_id} started discovery on {self.multicast_group}:{self.multicast_port}")
            print(
                f"[MulticastDiscovery] TCP listener at {self.listen_host}:{self.listen_port}")

        except Exception as e:
            print(f"[MulticastDiscovery] Failed to start: {e}")
            self._running = False
            if self._sock:
                try:
                    self._sock.close()
                except:
                    pass
                self._sock = None

    def stop(self):
        """Stop the discovery service."""
        if not self._running:
            return

        self._running = False
        self._shutdown.set()

        # Wait for threads
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2.0)
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        if self._expirer_thread and self._expirer_thread.is_alive():
            self._expirer_thread.join(timeout=2.0)

        # Close socket
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None

        print(
            f"[MulticastDiscovery] {self.kind.capitalize()} {self.node_id} stopped discovery")

    def _sender_loop(self):
        """Periodically broadcast HELLO messages."""
        hello_msg = {
            "type": "hello",
            "node_id": self.node_id,
            "kind": self.kind,
            "listen_host": self.listen_host,
            "listen_port": self.listen_port,
        }
        hello_data = json.dumps(hello_msg).encode('utf-8')
        address = (self.multicast_group, self.multicast_port)

        # Log first HELLO
        first_hello = True

        while self._running and not self._shutdown.is_set():
            try:
                self._sock.sendto(hello_data, address)
                if first_hello:
                    print(
                        f"[MulticastDiscovery] {self.kind.capitalize()} {self.node_id} sending first HELLO to {address}")
                    first_hello = False
            except Exception as e:
                if self._running:
                    print(f"[MulticastDiscovery] Error sending HELLO: {e}")

            # Wait for next interval
            self._shutdown.wait(self.hello_interval)

    def _listener_loop(self):
        """Listen for HELLO messages from other nodes."""
        while self._running and not self._shutdown.is_set():
            try:
                data, addr = self._sock.recvfrom(1024)
                try:
                    msg = json.loads(data.decode('utf-8'))
                    if msg.get("type") == "hello":
                        self._handle_hello(msg, addr)
                except (json.JSONDecodeError, KeyError) as e:
                    # Ignore malformed messages
                    pass
            except socket.timeout:
                # Expected - allows checking shutdown flag
                continue
            except Exception as e:
                if self._running:
                    print(f"[MulticastDiscovery] Error receiving HELLO: {e}")

    def _handle_hello(self, msg: dict, addr: Tuple[str, int]):
        """Handle a received HELLO message."""
        sender_id = msg.get("node_id")
        sender_kind = msg.get("kind")
        sender_host = msg.get("listen_host")
        sender_port = msg.get("listen_port")

        if not all([sender_id, sender_kind, sender_host, sender_port]):
            return

        # Ignore our own HELLO
        if sender_id == self.node_id:
            return

        same_kind_count = 0
        was_new = False
        with self.peers_lock:
            was_new = sender_id not in self.peers
            now = time.time()

            if was_new:
                # New peer discovered
                self.peers[sender_id] = PeerInfo(
                    node_id=sender_id,
                    kind=sender_kind,
                    host=sender_host,
                    port=sender_port,
                    last_seen=now,
                )
                print(
                    f"[MulticastDiscovery] Discovered new {sender_kind}: {sender_id} at {sender_host}:{sender_port}")
            else:
                # Update existing peer
                peer = self.peers[sender_id]
                peer.last_seen = now
                # Update host/port in case they changed
                peer.host = sender_host
                peer.port = sender_port

            # Count peers of the same kind while holding lock.
            # We pass this to callbacks so they don't need to re-acquire peers_lock.
            same_kind_count = sum(
                1 for peer in self.peers.values() if peer.kind == sender_kind)

        # Call callback outside the lock to avoid deadlock
        # NOTE: We call on_peer_discovered on *every* HELLO (not just the first time).
        # This allows higher-level managers to reconnect after restarts and to react
        # when the "number of environments" changes as stale entries expire.
        if self.on_peer_discovered:
            try:
                self.on_peer_discovered(
                    sender_id, sender_kind, sender_host, sender_port, same_kind_count)
            except Exception as e:
                print(
                    f"[MulticastDiscovery] Error in on_peer_discovered callback: {e}")

    def _expirer_loop(self):
        """Periodically check for expired peers."""
        while self._running and not self._shutdown.is_set():
            self._shutdown.wait(1.0)  # Check every second

            if self._shutdown.is_set():
                break

            now = time.time()
            expired: list[str] = []

            # Decide what's expired under lock
            with self.peers_lock:
                for node_id, peer in list(self.peers.items()):
                    if now - peer.last_seen > self.peer_timeout:
                        expired.append(node_id)

                # Remove expired peers under the same lock
                for node_id in expired:
                    if node_id in self.peers:
                        self.peers.pop(node_id, None)

            # Notify callbacks *outside* the lock to avoid deadlocks.
            for node_id in expired:
                print(
                    f"[MulticastDiscovery] Peer {node_id} expired (no HELLO for {self.peer_timeout}s)")
                if self.on_peer_expired:
                    try:
                        self.on_peer_expired(node_id)
                    except Exception as e:
                        print(
                            f"[MulticastDiscovery] Error in on_peer_expired callback: {e}")

    def get_peers(self) -> Dict[str, PeerInfo]:
        """Get a copy of the current peer table."""
        with self.peers_lock:
            return {node_id: PeerInfo(
                node_id=peer.node_id,
                kind=peer.kind,
                host=peer.host,
                port=peer.port,
                last_seen=peer.last_seen,
                tcp_conn=peer.tcp_conn,
            ) for node_id, peer in self.peers.items()}

    def get_peer(self, node_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        with self.peers_lock:
            peer = self.peers.get(node_id)
            if peer:
                return PeerInfo(
                    node_id=peer.node_id,
                    kind=peer.kind,
                    host=peer.host,
                    port=peer.port,
                    last_seen=peer.last_seen,
                    tcp_conn=peer.tcp_conn,
                )
            return None

    def set_peer_connection(self, node_id: str, tcp_conn: Optional[object]):
        """Set the TCP connection for a peer."""
        with self.peers_lock:
            if node_id in self.peers:
                self.peers[node_id].tcp_conn = tcp_conn

    def count_peers_by_kind(self, kind: str) -> int:
        """Count peers of a specific kind."""
        with self.peers_lock:
            return sum(1 for peer in self.peers.values() if peer.kind == kind)
