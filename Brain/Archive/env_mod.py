from brainprotocol import OBSERVATION, REWARD, ACTION, TERMINAL
from typing import Optional, Dict, Any, Tuple
from multiprocessing.connection import Client
import time


class SimpleEnv:
    """
    1D environment:
    state x, target x_target.
    brain action a in [-1, 1] changes x: x <- x + a * dt
    reward event can be sent on each step or only on special events.
    """

    def __init__(self, x0: float = 0.0, x_target: float = 5.0):
        self.x = x0
        self.x_target = x_target
        self.t = 0.0

    def reset(self) -> float:
        self.x = 0.0
        self.t = 0.0
        return self.x

    def step(self, action: float, dt: float) -> Tuple[float, float, bool, Dict[str, Any]]:
        # clip action
        if action > 1.0:
            action = 1.0
        if action < -1.0:
            action = -1.0

        self.t += dt
        self.x += action * dt

        # simple reward: negative distance
        reward = -abs(self.x - self.x_target)
        done = self.t >= 10.0

        info: Dict[str, Any] = {"t": self.t}
        return self.x, reward, done, info


def try_connect_brain(
    host: str = "localhost",
    port: int = 6000,
    authkey: bytes = b"brain-secret",
):
    try:
        conn = Client((host, port), authkey=authkey)
        print("[env] connected to brain")
        return conn
    except Exception:
        print("[env] no brain available, running standalone")
        return None


def run_env(
    host: str = "localhost",
    port: int = 6000,
    authkey: bytes = b"brain-secret",
    env_dt: float = 0.05,
):
    """
    Environment main loop.
    Runs even if no brain is available.
    Tries to connect on start and periodically if disconnected.
    """

    brain_conn: Optional[Client] = try_connect_brain(host, port, authkey)
    env = SimpleEnv()
    last_action = 0.0

    reconnect_interval = 2.0  # seconds
    last_reconnect_try = time.time()

    episode = 0

    while True:
        # if not connected, periodically retry
        if brain_conn is None and (time.time() - last_reconnect_try) >= reconnect_interval:
            last_reconnect_try = time.time()
            brain_conn = try_connect_brain(host, port, authkey)

        # start new episode
        episode += 1
        x = env.reset()
        done = False

        print(f"[env] starting episode {episode}")

        # send initial observation if connected
        if brain_conn is not None:
            try:
                brain_conn.send(
                    {
                        "type": OBSERVATION,
                        "sensors": [x],
                        "info": {"t": env.t, "episode": episode},
                    }
                )
            except (EOFError, OSError):
                print("[env] lost connection while sending initial observation")
                brain_conn = None

        while not done:
            # 1) if connected, pull any pending actions
            if brain_conn is not None:
                try:
                    while brain_conn.poll(0.0):
                        msg = brain_conn.recv()
                        if msg.get("type") == ACTION:
                            actions = msg.get("actions", [last_action])
                            if actions:
                                last_action = float(actions[0])
                except (EOFError, OSError):
                    print("[env] brain disconnected during episode")
                    brain_conn = None

            # 2) advance environment using latest action
            x, reward, done, info = env.step(last_action, env_dt)

            print(
                f"[env] t={info['t']:.2f}, x={x:.3f}, a={last_action:.2f}, r={reward:.3f}")

            # 3) send reward event, possibly without observation changes
            if brain_conn is not None:
                try:
                    # reward only
                    brain_conn.send(
                        {
                            "type": REWARD,
                            "value": float(reward),
                            "info": {"t": info["t"], "episode": episode},
                        }
                    )

                    # if you want, sometimes send observations too,
                    # for example every N steps or when state changes strongly
                    brain_conn.send(
                        {
                            "type": OBSERVATION,
                            "sensors": [x],
                            "info": {"t": info["t"], "episode": episode},
                        }
                    )
                except (EOFError, OSError):
                    print("[env] brain disconnected while sending reward/obs")
                    brain_conn = None

            # 4) sleep until next env tick
            time.sleep(env_dt)

        # end of episode event
        if brain_conn is not None:
            try:
                brain_conn.send(
                    {
                        "type": TERMINAL,
                        "info": {"t": env.t, "episode": episode},
                    }
                )
            except (EOFError, OSError):
                print("[env] brain disconnected while sending terminal")
                brain_conn = None
