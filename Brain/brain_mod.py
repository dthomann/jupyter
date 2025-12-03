
# brain_server.py
import time
from multiprocessing.connection import Listener
from typing import Optional

from brainprotocol import OBSERVATION, REWARD, ACTION, TERMINAL, SHUTDOWN


class Brain:
    """
    Minimal brain:
    - keeps last sensors and cumulative reward
    - runs its own internal tick
    - when it receives an OBSERVATION or REWARD event it decides an ACTION
    """

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.t = 0.0
        self.last_sensors = [0.0]
        self.cum_reward = 0.0
        self.last_action = 0.0
        self.pending_decision = False

    def on_observation(self, sensors, info):
        self.last_sensors = list(sensors)
        # mark that we should reconsider our action
        self.pending_decision = True

    def on_reward(self, value: float, info):
        self.cum_reward += value
        # reward alone can also trigger a decision
        self.pending_decision = True

    def tick(self, dt: float) -> Optional[dict]:
        """
        Called regularly. Returns an ACTION message dict when a new action
        should be sent, otherwise None.
        """
        self.t += dt

        if not self.pending_decision:
            return None

        # toy policy:
        # action depends on first sensor and cumulative reward
        s0 = self.last_sensors[0] if self.last_sensors else 0.0
        a = 1.0 if s0 + 0.1 * self.cum_reward >= 0.0 else -1.0

        self.last_action = a
        self.pending_decision = False

        return {
            "type": ACTION,
            "actions": [a],
            "info": {"t": self.t, "cum_reward": self.cum_reward},
        }


def run_brain_server(
    host: str = "localhost",
    port: int = 6000,
    authkey: bytes = b"brain-secret",
    dt: float = 0.02,
):
    """
    Long lived brain process.
    Listens once, accepts one environment at a time, and keeps running
    even if environments come and go.
    """
    address = (host, port)
    listener = Listener(address, authkey=authkey)
    print(f"[brain] listening on {address}")

    brain = Brain(dt=dt)

    while True:
        print("[brain] waiting for environment connection")
        conn = listener.accept()
        print(f"[brain] env connected from {listener.last_accepted}")

        try:
            running = True
            last_time = time.time()

            while running:
                now = time.time()
                elapsed = now - last_time
                if elapsed >= brain.dt:
                    last_time = now
                    action_msg = brain.tick(elapsed)
                    if action_msg is not None:
                        # send new action event
                        conn.send(action_msg)

                # non blocking check for incoming messages
                if conn.poll(0.0):
                    msg = conn.recv()
                    mtype = msg.get("type")

                    if mtype == OBSERVATION:
                        sensors = msg.get("sensors", [])
                        info = msg.get("info", {})
                        brain.on_observation(sensors, info)

                    elif mtype == REWARD:
                        value = float(msg.get("value", 0.0))
                        info = msg.get("info", {})
                        brain.on_reward(value, info)

                    elif mtype == TERMINAL:
                        # end of episode from env side
                        # brain could reset internal episode specific state here
                        pass

                    elif mtype == SHUTDOWN:
                        print("[brain] received shutdown message")
                        conn.close()
                        listener.close()
                        return

                # small sleep to avoid busy loop
                time.sleep(0.001)

        except (EOFError, OSError):
            print("[brain] environment disconnected unexpectedly")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[brain] connection closed, waiting for next env")
