# Running the Brain and Environment

This guide explains how to start, stop, and observe the brain and environment.

## Architecture

- **Environment Server** (`run_env_server.py`): Listens for connections and runs tic-tac-toe games
- **Brain Client** (`run_brain_client.py`): Connects to environment and learns to play

## Quick Start

### Option 1: Run Both Together (Recommended for Testing)

```bash
python run_both.py
```

This starts both the environment server and brain client in separate processes. Press `Ctrl+C` to stop both.

### Option 2: Run Separately (Recommended for Production)

**Terminal 1 - Start Environment Server:**
```bash
python run_env_server.py
```

**Terminal 2 - Start Brain Client:**
```bash
python run_brain_client.py
```

## Command Line Options

### Environment Server

```bash
python run_env_server.py
```

No options currently - uses defaults:
- Host: `localhost`
- Port: `6000`
- Authkey: `brain-secret`
- Tick interval: `0.05` seconds

### Brain Client

```bash
python run_brain_client.py [options]
```

Options:
- `--host HOST`: Environment server host (default: `localhost`)
- `--port PORT`: Environment server port (default: `6000`)
- `--authkey KEY`: Connection authkey (default: `brain-secret`)
- `--dt SECONDS`: Tick interval in seconds (default: `0.02`)
- `--lr RATE`: Learning rate (default: `0.001`)
- `--entropy COEFF`: Entropy coefficient (default: `0.001`)
- `--load PATH`: Load brain from saved file
- `--save PATH`: Save brain to file periodically
- `--save-every N`: Save every N episodes (default: `100`)

**Examples:**

Train a new brain:
```bash
python run_brain_client.py --lr 0.001 --entropy 0.001
```

Continue training from saved brain:
```bash
python run_brain_client.py --load brain_tictactoe_agent.pkl --save brain_tictactoe_agent.pkl --save-every 50
```

## Observing Behavior

### Console Output

Both processes print status messages:

**Environment Server:**
```
[env] listening on ('localhost', 6000)
[env] waiting for brain connection
[env] brain connected from ('127.0.0.1', 54321)
[env] starting episode 1
[env] t=0.00, x=[0, 0, 0, 0, 0, 0, 0, 0, 0], a=4, player=1, r=0.0, done=False
...
```

**Brain Client:**
```
[brain] connecting to environment server at ('localhost', 6000)
[brain] connected to environment server
```

### Monitoring Training Progress

The brain learns through self-play. You can observe:
- Episode numbers increasing
- Game outcomes (wins, losses, draws)
- Training happens automatically after each episode

### Saving and Loading

Save your trained brain:
```bash
python run_brain_client.py --save my_brain.pkl --save-every 100
```

Load and continue training:
```bash
python run_brain_client.py --load my_brain.pkl --save my_brain.pkl
```

## Stopping

### Graceful Shutdown

Press `Ctrl+C` in the terminal running the process. Both processes handle interrupts gracefully:
- Brain saves state if `--save` is specified
- Environment closes connections cleanly

### Force Stop

If needed, you can kill processes:
```bash
# Find processes
ps aux | grep -E "(run_env_server|run_brain_client)"

# Kill by PID
kill <PID>
```

## Troubleshooting

### "Connection refused" or "No brain available"

- Make sure environment server is running first
- Check that port 6000 is not in use: `lsof -i :6000`
- Verify host/port settings match

### Brain not learning

- Check that `episode_based_learning=True` (default)
- Verify learning rate is reasonable (try `--lr 0.001`)
- Ensure entropy coefficient is set (try `--entropy 0.001`)

### High CPU usage

- Increase `--dt` value (e.g., `--dt 0.05`) to reduce tick frequency
- This slows down the brain's decision-making loop

## Advanced Usage

### Multiple Brains

You can run multiple brain clients against the same environment (though they won't share learning):
```bash
# Terminal 1
python run_env_server.py

# Terminal 2
python run_brain_client.py --save brain1.pkl

# Terminal 3
python run_brain_client.py --save brain2.pkl
```

### Custom Configuration

Modify the scripts directly to change:
- Network settings (host, port, authkey)
- Brain architecture (hidden layers, activation functions)
- Training hyperparameters

See `run_brain_client.py` for brain configuration options.

