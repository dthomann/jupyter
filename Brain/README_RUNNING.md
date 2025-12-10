# Running the Brain and Environment

This guide explains how to start, stop, and observe the biologically-inspired brain and environment.

## Project Structure

**Main Scripts:**
- `run_brain_client.py` - Main script to start a brain client that connects to the environment and learns
- `run_env_server.py` - Main script to start the tic-tac-toe environment server that brains connect to
- `train_brain_tictactoe.py` - Standalone training script for training a brain on tic-tac-toe

**Core Modules:**
- `brain/` - Core brain implementation with biologically-inspired components (neuromodulators, memory, motivation, etc.)
- `brainprotocol.py` - Communication protocol constants for brain-environment messaging

**Environment:**
- `tic_tac_toe_env.py` - Tic-tac-toe environment implementation compatible with BrainAgent
- `tic_tac_toe_ui.py` - Command-line UI for playing tic-tac-toe against a trained brain
- `tic_tac_toe_gui.py` - Graphical UI for playing tic-tac-toe against a trained brain
- `tic_tac_toe_notebook.py` - Jupyter notebook-friendly interface for interactive tic-tac-toe games

**Directories:**
- `Brainstates/` - Default location for saved brain state files (.pkl)
- `Testing/` - Test files for validating brain components and integration
- `Archive/` - Deprecated or no longer relevant files
- `logs/` - Training metrics and logs (JSONL format)
- `checkpoints/` - Additional checkpoint files (legacy)

**Documentation:**
- `README_RUNNING.md` - This file: guide for running the brain and environment
- `HOW_NE_DA_AND_MEMORY_WORK.md` - Technical documentation on neuromodulators and memory systems
- `modfifybrain.md` - Documentation on modifying brain components

## Architecture

- **Environment Server** (`run_env_server.py`): Listens for connections and runs tic-tac-toe games
- **Brain Client** (`run_brain_client.py`): Connects to environment and learns to play

The brain is **biologically-inspired** and self-regulates learning through:
- **Neuromodulators**: Dopamine, Norepinephrine, Serotonin, Acetylcholine, and Cortisol dynamically modulate learning rates and exploration
- **Hippocampal Memory**: Experiences flow from short-term memory → episodic long-term memory based on salience
- **Intrinsic Motivation**: Curiosity, competence, and autonomy drive exploration
- **Homeostatic Drives**: Internal needs (energy, boredom) modulate reward values

**No manual hyperparameter tuning needed!** The brain adapts automatically to the environment.

## Quick Start

### Run Separately (Recommended)

**Terminal 1 - Start Environment Server:**
```bash
python run_env_server.py
```

**Terminal 2 - Start Brain Client:**
```bash
python run_brain_client.py
```

That's it! The brain will connect and start learning automatically. By default, it uses `Brainstates/brain_default.pkl` and can resume training on restart.

## Command Line Options

### Environment Server

```bash
python run_env_server.py [options]
```

**Options:**
- `--host HOST`: Server host (default: `localhost`)
- `--port PORT`: Server port (default: `6000`)
- `--authkey KEY`: Connection authkey (default: `brain-secret`)
- `--self-play`: Enable self-play mode (single brain plays against itself)
- `--training-mode`: Enable training mode - forces immediate wins/blocks (default: `True`)
- `--no-training-mode`: Disable training mode - allow all valid moves

**Examples:**

Two-brain mode (default - brains play against each other):
```bash
python run_env_server.py
```

Self-play mode (single brain):
```bash
python run_env_server.py --self-play
```

Custom port:
```bash
python run_env_server.py --port 7000
```

### Brain Client

```bash
python run_brain_client.py [options]
```

#### Essential Connection Parameters

- `--host HOST`: Environment server host (default: `localhost`)
- `--port PORT`: Environment server port (default: `6000`)
- `--authkey KEY`: Connection authkey (default: `brain-secret`)

#### Persistence

- `--load PATH`: Load brain from saved file (continues training from saved state). Defaults to `Brainstates/` folder if relative path.
- `--save PATH`: Save brain to file periodically (e.g., `brain.pkl`). Defaults to `Brainstates/` folder if relative path.
- `--name NAME`: Brain name for multiple simultaneous brains (creates `Brainstates/brain_{name}.pkl`). If not specified, uses `Brainstates/brain_default.pkl`
- `--save-every N`: Save every N episodes (default: `500`)

**Note:** All brain state files are saved/loaded from the `Brainstates/` folder by default. You can use absolute paths to save elsewhere if needed.

#### Training Control

- `--max-episodes N`: Maximum episodes to train (default: unlimited, Ctrl+C to stop)
- `--seed N`: Random seed for reproducibility (default: `42`, use `-1` for random)
- `--play-against-opponent`: Play against another brain (disable self-play random moves). Use with two-brain server mode.

#### Bi-directional Communication (Advanced)

- `--brain-id ID`: Unique brain identifier (default: auto-generated)
- `--enable-listener`: Enable incoming connection listener
- `--listen-host HOST`: Host to listen on for incoming connections (requires `--enable-listener`)
- `--listen-port PORT`: Port to listen on for incoming connections (requires `--listen-host`)
- `--peer PEER_ID HOST PORT`: Add outgoing peer connection (can be specified multiple times)

#### Optional: Biological Parameters (Rarely Need Adjustment)

- `--curiosity SCALE`: Curiosity scale (default: auto, based on environment)
- `--competence SCALE`: Competence scale (default: auto)

#### Optional: Monitoring/Logging

- `--stats-every N`: Print stats every N episodes (default: `100`)
- `--metrics-log PATH`: Log metrics to JSONL file (default: `logs/brain_metrics.jsonl`, use `"none"` to disable)

**Examples:**

Basic usage (uses `Brainstates/brain_default.pkl`, can resume on restart):
```bash
python run_brain_client.py
```

Save periodically during training (saves to `Brainstates/brain.pkl`):
```bash
python run_brain_client.py --save brain.pkl
```

Save every 100 episodes instead of default 500:
```bash
python run_brain_client.py --save brain.pkl --save-every 100
```

Continue training from saved brain (loads from `Brainstates/brain.pkl`):
```bash
python run_brain_client.py --load brain.pkl --save brain.pkl
```

Use absolute path to save elsewhere:
```bash
python run_brain_client.py --save /path/to/custom/location/brain.pkl
```

Train for exactly 1000 episodes:
```bash
python run_brain_client.py --max-episodes 1000
```

Play against another brain (two-brain mode):
```bash
# Terminal 1: Start server (default two-brain mode)
python run_env_server.py

# Terminal 2: Start first brain
python run_brain_client.py --name player1 --play-against-opponent

# Terminal 3: Start second brain
python run_brain_client.py --name player2 --play-against-opponent
```

Connect to different host/port:
```bash
python run_brain_client.py --host 192.168.1.100 --port 7000
```

Enable bidirectional communication (brain accepts incoming connections):
```bash
python run_brain_client.py --enable-listener --listen-host localhost --listen-port 7000
```

Connect to additional peers (besides environment):
```bash
python run_brain_client.py --peer other_brain localhost 7001
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
🧠 Biologically-Inspired Brain Client
======================================================================
Learning through:
  • Neuromodulators (DA, NE, 5-HT, ACh, Cortisol)
  • Hippocampal memory (STM → Episodic consolidation)
  • Intrinsic motivation (Curiosity, Competence, Autonomy)
  • Homeostatic drives (Energy, Boredom)
======================================================================
[brain] connecting to environment server at ('localhost', 6000)
[brain] connected to environment server
```

### Monitoring Training Progress

The brain learns through self-play or against opponents. You can observe:
- Episode numbers increasing
- Game outcomes (wins, losses, draws)
- Neuromodulator levels (NE, DA)
- Learning metrics (policy loss, value loss)
- Performance against random opponents (periodically)

Example output:
```
Ep: 100 | 200: W=45 L=38 D=117 (59%) | Ent=0.0008 | LR=0.00095 | NE=0.12 DA=0.48
```

The brain learns continuously - no need to tune learning rates or exploration schedules!

### Saving and Loading

**Default Behavior:**
- If no `--load` or `--save` is specified, the brain uses `Brainstates/brain_default.pkl`
- All brain state files are saved/loaded from the `Brainstates/` folder by default
- The brain auto-saves on exit (Ctrl+C)
- You can resume training by simply running the same command again

**Explicit Save/Load:**
```bash
# Save periodically (saves to Brainstates/my_brain.pkl)
python run_brain_client.py --save my_brain.pkl --save-every 100

# Load and continue training (from Brainstates/my_brain.pkl)
python run_brain_client.py --load my_brain.pkl --save my_brain.pkl
```

**Multiple Brains:**
```bash
# Use named brains for multiple simultaneous instances
# Files saved to Brainstates/brain_player1.pkl and Brainstates/brain_player2.pkl
python run_brain_client.py --name player1 --save brain_player1.pkl
python run_brain_client.py --name player2 --save brain_player2.pkl
```

## Stopping

### Graceful Shutdown

Press `Ctrl+C` in the terminal running the process. Both processes handle interrupts gracefully:
- Brain auto-saves state (to `Brainstates/brain_default.pkl` or specified `--save` file)
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
- Verify host/port settings match between server and client

### Brain not learning

- Check that `episode_based_learning=True` (default)
- The brain self-regulates learning through neuromodulators - no manual tuning needed
- Ensure you're giving it enough episodes to learn (try `--max-episodes 10000`)

### High CPU usage

- The brain uses a default tick interval of 0.02 seconds
- This is optimized for learning - reducing it may slow down learning

### Two-brain mode not working

- Make sure environment server is running in default mode (not `--self-play`)
- Both brain clients should use `--play-against-opponent` flag
- Use `--name` to give each brain a unique identifier

## Advanced Usage

### Multiple Brains

You can run multiple brain clients against the same environment:

```bash
# Terminal 1: Start server (two-brain mode)
python run_env_server.py

# Terminal 2: First brain (saves to Brainstates/brain_player1.pkl)
python run_brain_client.py --name player1 --play-against-opponent --save brain_player1.pkl

# Terminal 3: Second brain (saves to Brainstates/brain_player2.pkl)
python run_brain_client.py --name player2 --play-against-opponent --save brain_player2.pkl
```

### Custom Configuration

Modify the scripts directly to change:
- Network settings (host, port, authkey)
- Brain architecture (hidden layers, activation functions)
- Training hyperparameters (though neuromodulators handle most adaptation automatically)

See `run_brain_client.py` for brain configuration options.

### Metrics Logging

The brain logs detailed metrics to a JSONL file by default:
```bash
# Default location
python run_brain_client.py  # Logs to logs/brain_metrics.jsonl

# Custom location
python run_brain_client.py --metrics-log my_metrics.jsonl

# Disable logging
python run_brain_client.py --metrics-log none
```

### Reproducibility

Use the `--seed` option for reproducible training:
```bash
python run_brain_client.py --seed 42
```

Use `--seed -1` for random (non-reproducible) training.
