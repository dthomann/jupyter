"""
Notebook-friendly Tic Tac Toe game interface.
Can be used interactively in Jupyter notebooks.
"""

import numpy as np
import os
import time
from tic_tac_toe_env import TicTacToeEnv
from brain import BrainAgent

# Try to import widgets (optional)
try:
    from IPython.display import display, clear_output
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None


class TicTacToeNotebook:
    """Notebook-friendly Tic Tac Toe UI using widgets."""

    def __init__(self, agent_path=None, save_path='tictactoe_agent.pkl'):
        self.env = TicTacToeEnv()
        self.save_path = save_path

        # Initialize agent
        if agent_path and os.path.exists(agent_path):
            print(f"Loading agent from {agent_path}...")
            self.agent = BrainAgent.load(agent_path)
            # Disable intrinsic motivation for Tic Tac Toe (it adds noise)
            self.agent.intrinsic.curiosity_scale = 0.0
            self.agent.intrinsic.learning_progress_scale = 0.0
            print(f"Loaded agent with {self.agent.global_step} training steps")
        else:
            print("Creating new agent...")
            self.agent = BrainAgent(
                obs_dim=9,
                latent_dims=[32, 16],
                n_actions=9,
                lr_model=1e-3,
                lr_policy=1e-2,  # Learning rate matching test_training.py
                replay_batch_size=32,
                use_raw_obs_for_policy=True,  # Use raw board state for stable policy learning
            )
            # Disable intrinsic motivation for Tic Tac Toe (it adds noise)
            self.agent.intrinsic.curiosity_scale = 0.0
            self.agent.intrinsic.learning_progress_scale = 0.0
            print("New agent created")

        self.stats = {
            'games_played': 0,
            'agent_wins': 0,
            'human_wins': 0,
            'draws': 0,
        }

        self.current_game_state = None
        self.agent_turn = True
        self.game_active = False
        self.continuous_mode = False
        self.training_mode = False
        self.should_exit = False
        self.training_stats = {
            'training_games': 0,
            'x_wins': 0,
            'o_wins': 0,
            'draws': 0,
        }
        self.training_history = []  # Track win rates over time

    def print_board(self, board=None):
        """Display the board."""
        if board is None:
            board = self.env.board
        board_2d = board.reshape(3, 3)
        symbols = {1: 'X', -1: 'O', 0: ' '}

        print("\n" + "="*30)
        print("Current Board (Agent=X, You=O)")
        print("="*30)
        print("\n  0   1   2")
        for i in range(3):
            row_str = f"{i} "
            for j in range(3):
                val = board_2d[i, j]
                pos = i * 3 + j
                if val == 0:
                    row_str += f" {pos} "
                else:
                    row_str += f" {symbols[val]} "
                if j < 2:
                    row_str += "|"
            print(row_str)
            if i < 2:
                print("  -----------")
        print()

    def make_agent_move(self, temperature=None, greedy=None):
        """Make agent's move and return the result."""
        if not self.game_active or not self.agent_turn:
            return None

        obs = self.env._get_obs()

        # For human play, use learned policy deterministically after training
        # Use greedy if agent has been trained, otherwise use low temperature
        if greedy is None:
            # Use greedy mode if agent has significant training (same threshold as test)
            greedy = self.agent.global_step > 1000

        if greedy:
            # Greedy mode: always pick best action from learned policy
            temperature = 0.0
        elif temperature is None:
            # Use very low temperature for exploitation (not exploration)
            temperature = 0.05  # Low temperature for near-deterministic play

        action, z, value, x = self.agent.act(
            obs, temperature=temperature, greedy=greedy)

        # If agent chooses invalid move, pick random valid one
        if not self.env._is_valid_move(action):
            valid_actions = self.env.get_valid_actions()
            if valid_actions:
                action = self.env.rng.choice(valid_actions)
            else:
                return None

        next_obs, reward, done, info = self.env.step(action)

        if done:
            if self.env.winner == 'X':
                self.stats['agent_wins'] += 1
                agent_reward = 1.0  # Reward wins
            elif self.env.winner == 'draw':
                self.stats['draws'] += 1
                agent_reward = 0.0  # Draws are neutral
            else:
                agent_reward = 0.0

            # Use very low learning rate when playing humans to preserve trained policy
            # Or disable learning entirely if agent is well-trained
            if self.agent.global_step > 5000:
                # Well-trained agent: don't learn from human games (preserve policy)
                pass
            else:
                # Still learning: use reduced learning rate
                original_lr = self.agent.lr_policy
                self.agent.lr_policy = original_lr * 0.1  # 10% of normal learning rate
                self.agent.online_update(
                    obs=obs,
                    x=x,
                    z=z,
                    action=action,
                    external_reward=agent_reward,
                    next_obs=next_obs,
                    done=True,
                )
                self.agent.lr_policy = original_lr  # Restore
            self.game_active = False
            return {'action': action, 'done': True, 'winner': self.env.winner}

        # Store for learning after human responds
        self.current_game_state = {
            'prev_obs': obs.copy(),
            'prev_z': z,
            'prev_x': x,
            'prev_action': action,
        }

        self.agent_turn = False
        return {'action': action, 'done': False}

    def make_human_move(self, action):
        """Make human's move and return the result."""
        if not self.game_active or self.agent_turn:
            return None

        if not self.env._is_valid_move(action):
            return {'error': 'Invalid move'}

        next_obs, reward, done, info = self.env.make_opponent_move(action)

        if done:
            if self.env.winner == 'O':
                self.stats['human_wins'] += 1
                agent_reward = -1.0  # Penalize losses
            elif self.env.winner == 'draw':
                self.stats['draws'] += 1
                agent_reward = 0.0  # Neutral reward for draws
            else:
                agent_reward = 0.0

            # Learn from final state (only if agent not well-trained)
            if self.current_game_state is not None:
                if self.agent.global_step > 5000:
                    # Well-trained: don't learn from human games
                    pass
                else:
                    # Still learning: use reduced learning rate
                    original_lr = self.agent.lr_policy
                    self.agent.lr_policy = original_lr * 0.1
                    self.agent.online_update(
                        obs=self.current_game_state['prev_obs'],
                        x=self.current_game_state['prev_x'],
                        z=self.current_game_state['prev_z'],
                        action=self.current_game_state['prev_action'],
                        external_reward=agent_reward,
                        next_obs=next_obs,
                        done=True,
                    )
                    self.agent.lr_policy = original_lr
            self.game_active = False
            return {'done': True, 'winner': self.env.winner}

        # Learn from transition (only if agent not well-trained)
        if self.current_game_state is not None:
            if self.agent.global_step <= 5000:
                # Still learning: use reduced learning rate
                original_lr = self.agent.lr_policy
                self.agent.lr_policy = original_lr * 0.1
                self.agent.online_update(
                    obs=self.current_game_state['prev_obs'],
                    x=self.current_game_state['prev_x'],
                    z=self.current_game_state['prev_z'],
                    action=self.current_game_state['prev_action'],
                    external_reward=0.0,
                    next_obs=next_obs,
                    done=False,
                )
                self.agent.lr_policy = original_lr

        self.agent_turn = True
        return {'done': False}

    def start_game(self, agent_first=True):
        """Start a new game."""
        obs = self.env.reset()
        self.stats['games_played'] += 1
        self.game_active = True
        self.agent_turn = agent_first
        self.current_game_state = None

        print(f"\n{'='*50}")
        print(f"Game #{self.stats['games_played']}")
        print(f"{'='*50}")
        if agent_first:
            print("Agent goes first (X)")
        else:
            print("You go first (O)")

        self.print_board()

        if agent_first:
            result = self.make_agent_move()
            if result:
                self.print_board()
                if result.get('done'):
                    if result.get('winner') == 'X':
                        print("🤖 Agent wins!")
                    elif result.get('winner') == 'draw':
                        print("🤝 It's a draw!")
                else:
                    print(f"Agent plays position {result['action']}")
                    print("Your turn!")

    def print_stats(self):
        """Print game statistics."""
        print("\n" + "="*50)
        print("Statistics")
        print("="*50)
        print(f"Games played: {self.stats['games_played']}")
        if self.stats['games_played'] > 0:
            print(
                f"Agent wins: {self.stats['agent_wins']} ({100*self.stats['agent_wins']/self.stats['games_played']:.1f}%)")
            print(
                f"Human wins: {self.stats['human_wins']} ({100*self.stats['human_wins']/self.stats['games_played']:.1f}%)")
            print(
                f"Draws: {self.stats['draws']} ({100*self.stats['draws']/self.stats['games_played']:.1f}%)")
        print(f"Agent training steps: {self.agent.global_step}")
        print("="*50)

    def save_agent(self):
        """Save the agent state."""
        self.agent.save(self.save_path)
        print(f"\nAgent saved to {self.save_path}")

    def create_widget_ui(self):
        """Create an interactive widget-based UI."""
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets not available. Install with: pip install ipywidgets")

        # Create buttons for each position
        self.position_buttons = []
        button_layout = widgets.Layout(width='50px', height='50px')

        for i in range(9):
            btn = widgets.Button(
                description=str(i),
                layout=button_layout,
                disabled=False
            )
            btn.position = i
            btn.on_click(self._on_button_click)
            self.position_buttons.append(btn)

        # Create control buttons
        self.new_game_btn = widgets.Button(
            description='New Game (Agent First)', button_style='primary')
        self.new_game_btn.on_click(
            lambda b: self._start_new_game(agent_first=True))

        self.new_game_human_btn = widgets.Button(
            description='New Game (You First)', button_style='primary')
        self.new_game_human_btn.on_click(
            lambda b: self._start_new_game(agent_first=False))

        self.continuous_btn = widgets.Button(
            description='Start Continuous Play', button_style='success')
        self.continuous_btn.on_click(lambda b: self._toggle_continuous())

        self.train_btn = widgets.Button(
            description='Start Training', button_style='info')
        self.train_btn.on_click(lambda b: self._toggle_training())

        self.exit_btn = widgets.Button(
            description='Exit', button_style='danger')
        self.exit_btn.on_click(lambda b: self._exit_game())

        self.stats_btn = widgets.Button(description='Show Stats')
        self.stats_btn.on_click(lambda b: self._show_stats())

        self.save_btn = widgets.Button(description='Save Agent')
        self.save_btn.on_click(lambda b: self._save_agent())

        # Status output
        self.status_output = widgets.Output()

        # Training progress output (separate for better visibility)
        self.training_output = widgets.Output()

        # Layout
        board_grid = widgets.GridBox(
            self.position_buttons,
            layout=widgets.Layout(
                grid_template_columns='repeat(3, 60px)',
                grid_gap='5px'
            )
        )

        controls = widgets.HBox([
            self.new_game_btn,
            self.new_game_human_btn,
            self.continuous_btn,
            self.train_btn,
            self.exit_btn,
            self.stats_btn,
            self.save_btn
        ])

        ui = widgets.VBox([
            widgets.HTML(
                "<h2>Tic Tac Toe - Play against a Learning Agent!</h2>"),
            controls,
            board_grid,
            widgets.HTML("<h3>Game Status:</h3>"),
            self.status_output,
            widgets.HTML("<h3>Training Progress:</h3>"),
            self.training_output
        ])

        return ui

    def _on_button_click(self, button):
        """Handle button click for making a move."""
        with self.status_output:
            clear_output(wait=True)

            if not self.game_active:
                print("Start a new game first!")
                return

            if self.agent_turn:
                print("It's the agent's turn! Click 'New Game' to start.")
                return

            action = button.position
            result = self.make_human_move(action)

            if result is None:
                print("Game not active or not your turn!")
                return

            if result.get('error'):
                print(f"Error: {result['error']}")
                return

            self.print_board()

            if result.get('done'):
                winner = result.get('winner')
                if winner == 'O':
                    print("🎉 You win!")
                elif winner == 'draw':
                    print("🤝 It's a draw!")
                self._update_button_states()
                self._handle_game_end()
                return

            # Agent's turn
            print("Agent's turn...")
            agent_result = self.make_agent_move()

            if agent_result:
                self.print_board()
                if agent_result.get('done'):
                    winner = agent_result.get('winner')
                    if winner == 'X':
                        print("🤖 Agent wins!")
                    elif winner == 'draw':
                        print("🤝 It's a draw!")
                    self._handle_game_end()
                else:
                    print(f"Agent plays position {agent_result['action']}")
                    print("Your turn!")

            self._update_button_states()

    def _update_button_states(self):
        """Update button states based on game state."""
        valid_actions = self.env.get_valid_actions() if self.game_active else []

        for i, btn in enumerate(self.position_buttons):
            board_val = self.env.board[i] if self.game_active else 0
            if board_val == 1:  # Agent (X)
                btn.description = 'X'
                btn.button_style = 'info'
                btn.disabled = True
            elif board_val == -1:  # Human (O)
                btn.description = 'O'
                btn.button_style = 'warning'
                btn.disabled = True
            else:
                btn.description = str(i)
                btn.button_style = ''
                btn.disabled = not (
                    self.game_active and i in valid_actions and not self.agent_turn)

    def _start_new_game(self, agent_first=True):
        """Start a new game (for widget UI)."""
        with self.status_output:
            clear_output(wait=True)
            self.start_game(agent_first=agent_first)
            self._update_button_states()

    def _toggle_continuous(self):
        """Toggle continuous play mode."""
        with self.status_output:
            clear_output(wait=True)
            if not self.continuous_mode:
                self.continuous_mode = True
                self.should_exit = False
                self.continuous_btn.description = 'Stop Continuous Play'
                self.continuous_btn.button_style = 'warning'
                print("Continuous play mode started!")
                print(
                    "Games will automatically restart. Click 'Stop Continuous Play' to exit.")
                # Start first game
                if not self.game_active:
                    self._start_new_game(agent_first=True)
            else:
                self.continuous_mode = False
                self.continuous_btn.description = 'Start Continuous Play'
                self.continuous_btn.button_style = 'success'
                print("Continuous play mode stopped.")

    def _exit_game(self):
        """Exit the game and save agent."""
        with self.status_output:
            clear_output(wait=True)
            self.should_exit = True
            self.continuous_mode = False
            self.save_agent()
            self.print_stats()
            print("\nThanks for playing!")

    def _show_stats(self):
        """Show stats in widget output."""
        with self.status_output:
            clear_output(wait=True)
            self.print_stats()

    def _save_agent(self):
        """Save agent in widget output."""
        with self.status_output:
            clear_output(wait=True)
            self.save_agent()

    def _flip_board_perspective(self, obs):
        """Flip board perspective: X becomes O and vice versa."""
        return -obs

    def _play_self_game(self, temperature=None):
        """Play a game where the agent plays against itself."""
        obs = self.env.reset()
        self.training_stats['training_games'] += 1
        game_history = []  # Store (obs, action, z, x) for learning

        # Adaptive temperature: start high for exploration, decrease over time
        # Match test_training.py schedule
        if temperature is None:
            base_temp = 3.0  # Higher initial temperature for exploration
            decay = 0.9998   # Decay rate matching test_training.py
            temperature = base_temp * (decay ** self.agent.global_step)
            # Lower minimum for better exploitation
            temperature = max(0.05, temperature)

        # Randomly choose who goes first
        agent_is_x = np.random.choice([True, False])

        while not self.env.done:
            # Flip board for O's perspective so both players see their pieces as +1
            if agent_is_x:
                current_obs = obs.copy()
            else:
                current_obs = -obs.copy()  # Flip: O sees its pieces as +1

            # Get valid actions first, then select from them
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break

            # Agent chooses action from valid actions only
            _, z, value, x = self.agent.act(
                current_obs, temperature=temperature, greedy=False)
            # Get Q-values/logits for policy (include bias term)
            if self.agent.use_raw_obs_for_policy:
                policy_state = x
            else:
                policy_state = z
            logits = self.agent.actor_critic.policy_logits(policy_state)

            # Mask invalid actions by setting their logits to very negative value
            masked_logits = logits.copy()
            for i in range(9):
                if i not in valid_actions:
                    masked_logits[i] = -1e10

            # Sample from valid actions only
            if temperature <= 0:
                action = int(np.argmax(masked_logits))
            else:
                # Softmax over masked logits
                exp_logits = np.exp(masked_logits / temperature)
                probs = exp_logits / exp_logits.sum()
                # Sample only from valid actions
                valid_probs = probs[valid_actions]
                valid_probs = valid_probs / valid_probs.sum()
                action = valid_actions[self.agent.rng.choice(
                    len(valid_actions), p=valid_probs)]

            # Make move and collect intermediate rewards
            if agent_is_x:
                next_obs, step_reward, done, info = self.env.step(action)
            else:
                next_obs, step_reward, done, info = self.env.make_opponent_move(
                    action)
                step_reward = -step_reward  # Flip reward for O perspective

            # Store state for learning with intermediate reward
            game_history.append({
                'obs': current_obs.copy(),
                'z': z,
                'x': x,
                'action': action,
                'is_x': agent_is_x,
                'intermediate_reward': step_reward  # Store intermediate reward
            })

            if done:
                # Determine rewards - optimize for wins, avoid losses
                if self.env.winner == 'X':
                    x_reward = 1.0   # Reward wins as X
                    o_reward = -1.0  # Penalize losses as O
                    self.training_stats['x_wins'] += 1
                elif self.env.winner == 'O':
                    x_reward = -1.0  # Penalize losses as X
                    o_reward = 1.0   # Reward wins as O
                    self.training_stats['o_wins'] += 1
                elif self.env.winner == 'draw':
                    x_reward = 0.0   # Draws are neutral
                    o_reward = 0.0
                    self.training_stats['draws'] += 1
                else:
                    x_reward = 0.0
                    o_reward = 0.0

                # Track win rate history
                if self.training_stats['training_games'] % 100 == 0:
                    total = self.training_stats['training_games']
                    x_win_rate = self.training_stats['x_wins'] / \
                        total if total > 0 else 0
                    o_win_rate = self.training_stats['o_wins'] / \
                        total if total > 0 else 0
                    draw_rate = self.training_stats['draws'] / \
                        total if total > 0 else 0
                    self.training_history.append({
                        'games': total,
                        'steps': self.agent.global_step,
                        'x_win_rate': x_win_rate,
                        'o_win_rate': o_win_rate,
                        'draw_rate': draw_rate,
                    })

                # CRITICAL: Use Monte Carlo returns for all moves
                # Each move gets the final reward discounted by remaining steps
                # This propagates the reward signal effectively
                gamma = 0.99
                n_moves = len(game_history)
                final_obs = self.env._get_obs()

                # Update all moves with their Monte Carlo returns
                # Update in reverse order (last move first) for better learning
                for i in range(n_moves - 1, -1, -1):
                    move_data = game_history[i]
                    steps_remaining = n_moves - 1 - i  # Steps from this move to end

                    # Determine which player's reward to use and compute Monte Carlo return
                    if move_data['is_x']:
                        move_reward = x_reward * (gamma ** steps_remaining)
                    else:
                        move_reward = o_reward * (gamma ** steps_remaining)

                    # Use final observation as-is (no flipping)
                    final_obs_for_move = final_obs

                    # For Monte Carlo, update policy/critic directly without world model updates
                    # The world model should only learn from actual step-by-step transitions
                    z_next, x_next = self.agent.encode_state(
                        final_obs_for_move)
                    policy_state = move_data['x'] if self.agent.use_raw_obs_for_policy else move_data['z']
                    policy_next_state = x_next if self.agent.use_raw_obs_for_policy else z_next

                    # Direct actor-critic update (bypass world model learning for MC)
                    # Use constant learning rate for stability (don't modulate by neuromodulators)
                    # Store original neuromodulator values to restore after update
                    original_dopamine = self.agent.neuromodulators.dopamine
                    original_norepinephrine = self.agent.neuromodulators.norepinephrine

                    # Temporarily set neuromodulators to zero to get constant learning rate
                    self.agent.neuromodulators.dopamine = 0.0
                    self.agent.neuromodulators.norepinephrine = 0.0

                    td_error, value, next_value = self.agent.actor_critic.update(
                        state=policy_state,
                        action=move_data['action'],
                        # External reward only (intrinsic disabled)
                        reward=move_reward,
                        next_state=policy_next_state,
                        done=True,
                        neuromodulators=self.agent.neuromodulators,
                        base_lr=self.agent.lr_policy,
                    )

                    # Restore neuromodulator values (they're used for tracking, not just LR modulation)
                    self.agent.neuromodulators.dopamine = original_dopamine
                    self.agent.neuromodulators.norepinephrine = original_norepinephrine

                    # Update neuromodulators
                    self.agent.neuromodulators.update(
                        reward=move_reward,
                        value=value,
                        next_value=next_value,
                        pred_error_norm=0.0,  # No prediction error for MC updates
                    )

                    # Store transition for offline replay (but don't update world model)
                    self.agent.memory.store(
                        move_data['x'], move_data['z'], move_data['action'],
                        move_reward, x_next, z_next, True
                    )

                    self.agent.global_step += 1

                break

            obs = next_obs
            agent_is_x = not agent_is_x

        # Periodic offline learning - replay past experiences
        if self.agent.global_step % 20 == 0 and len(self.agent.memory.buffer) > 0:
            # More batches for better learning
            self.agent.offline_replay(n_batches=10)

    def _toggle_training(self):
        """Toggle self-play training mode."""
        with self.status_output:
            clear_output(wait=True)
            if not self.training_mode:
                self.training_mode = True
                self.should_exit = False
                self.continuous_mode = False  # Disable continuous play when training
                self.train_btn.description = 'Stop Training'
                self.train_btn.button_style = 'warning'
                self.continuous_btn.description = 'Start Continuous Play'
                self.continuous_btn.button_style = 'success'
                print("Training mode started!")
                print("Agent is playing against itself to improve...")
                print("Click 'Stop Training' to exit.")

                # Initialize training output
                with self.training_output:
                    clear_output(wait=True)
                    print("Initializing training...")
                    print(f"Starting training steps: {self.agent.global_step}")
                    print(
                        f"Training games so far: {self.training_stats['training_games']}")

                # Start training loop
                self._training_loop()
            else:
                self.training_mode = False
                self.train_btn.description = 'Start Training'
                self.train_btn.button_style = 'info'
                print("Training mode stopped.")
                with self.training_output:
                    clear_output(wait=True)
                    self._print_training_stats()

    def _update_training_display(self, game_count, last_step, last_games):
        """Update training display."""
        try:
            with self.training_output:
                clear_output(wait=True)
                steps_this_batch = self.agent.global_step - last_step
                games_this_batch = self.training_stats['training_games'] - last_games
                print("Training Progress:")
                print(f"  Games completed: {game_count}")
                print(f"  Games this batch: {games_this_batch}")
                print(f"  Training steps this batch: {steps_this_batch}")
                print(f"  Total training steps: {self.agent.global_step}")
                print()
                self._print_training_stats()

                # Show recent trend if available
                if len(self.training_history) >= 2:
                    recent = self.training_history[-1]
                    prev = self.training_history[-2]
                    print(f"\nTrend (last 100 games):")
                    print(
                        f"  X win rate: {prev['x_win_rate']:.1%} -> {recent['x_win_rate']:.1%}")
                    print(
                        f"  O win rate: {prev['o_win_rate']:.1%} -> {recent['o_win_rate']:.1%}")

                # Show current temperature
                base_temp = 2.0
                decay = 0.9999
                current_temp = base_temp * (decay ** self.agent.global_step)
                current_temp = max(0.2, current_temp)
                print(f"\nCurrent exploration temperature: {current_temp:.3f}")
        except Exception as e:
            # Fallback to print if widget update fails
            print(f"Training update error: {e}")

    def _training_loop(self):
        """Run training loop (non-blocking for widget UI)."""
        import threading

        def train():
            game_count = 0
            last_step = self.agent.global_step
            last_games = self.training_stats['training_games']

            # Show initial state
            self._update_training_display(0, last_step, last_games)

            while self.training_mode and not self.should_exit:
                # Use adaptive temperature
                self._play_self_game(temperature=None)
                game_count += 1

                # Update stats display frequently
                update_freq = 1 if game_count <= 50 else (
                    5 if game_count <= 500 else 10)
                if game_count % update_freq == 0:
                    self._update_training_display(
                        game_count, last_step, last_games)
                    last_step = self.agent.global_step
                    last_games = self.training_stats['training_games']

                    # Small sleep to allow UI updates
                    time.sleep(0.05)

        # Run training in background thread
        self.training_thread = threading.Thread(target=train, daemon=True)
        self.training_thread.start()

    def _print_training_stats(self):
        """Print training statistics."""
        if self.training_stats['training_games'] > 0:
            print(f"Win rates:")
            print(
                f"  X (first player): {self.training_stats['x_wins']} ({100*self.training_stats['x_wins']/self.training_stats['training_games']:.1f}%)")
            print(
                f"  O (second player): {self.training_stats['o_wins']} ({100*self.training_stats['o_wins']/self.training_stats['training_games']:.1f}%)")
            print(
                f"  Draws: {self.training_stats['draws']} ({100*self.training_stats['draws']/self.training_stats['training_games']:.1f}%)")
            print()
            print(f"Agent state:")
            print(f"  Total training steps: {self.agent.global_step}")
            print(f"  Memory size: {len(self.agent.memory.buffer)}")
            print(f"  Dopamine: {self.agent.neuromodulators.dopamine:.3f}")
            print(
                f"  Curiosity drive: {self.agent.drives.curiosity_drive:.3f}")
        else:
            print("No training games completed yet.")

    def _handle_game_end(self):
        """Handle game end - auto-start next game if in continuous mode."""
        # Periodic offline learning
        if self.agent.global_step % 20 == 0 and len(self.agent.memory.buffer) > 0:
            self.agent.offline_replay(n_batches=10)

        if self.continuous_mode and not self.should_exit:
            # Auto-start next game with random starting player
            time.sleep(1)  # Brief pause
            agent_first = np.random.choice([True, False])
            self._start_new_game(agent_first=agent_first)

    def play_simple(self, agent_first=True, continuous=True):
        """
        Simple text-based play function for notebooks.
        Use this if widgets don't work.

        Args:
            agent_first: Whether agent goes first
            continuous: If True, keeps playing until user types 'q' to quit
        """
        print("\n" + "="*60)
        print("Tic Tac Toe - Play against a Learning Agent!")
        print("="*60)
        print("Type 'q' at any time to quit")
        print("="*60)

        while True:
            # In continuous mode, randomly choose who goes first each game
            if continuous:
                agent_first = np.random.choice([True, False])

            self.start_game(agent_first=agent_first)

            while self.game_active:
                if self.agent_turn:
                    result = self.make_agent_move()
                    if result:
                        self.print_board()
                        if result.get('done'):
                            break
                        print(f"Agent plays position {result['action']}")
                else:
                    valid_actions = self.env.get_valid_actions()
                    print(f"Valid positions: {valid_actions}")
                    try:
                        move_str = input(
                            "Enter your move (0-8, or 'q' to quit): ").strip()
                        if move_str.lower() == 'q':
                            self.should_exit = True
                            break
                        move = int(move_str)
                        result = self.make_human_move(move)
                        if result:
                            self.print_board()
                            if result.get('done'):
                                break
                            if result.get('error'):
                                print(result['error'])
                                continue
                    except ValueError:
                        print("Please enter a number between 0-8 or 'q' to quit.")
                        continue
                    except KeyboardInterrupt:
                        print("\nQuitting...")
                        self.should_exit = True
                        break

            if self.should_exit:
                break

            # Periodic offline learning
            if self.agent.global_step % 20 == 0 and len(self.agent.memory.buffer) > 0:
                print("Agent is reviewing past games...")
                self.agent.offline_replay(n_batches=10)

            self.print_stats()

            if not continuous:
                break

            # Ask to continue
            try:
                cont = input(
                    "\nPlay again? (y/n, default=y): ").strip().lower()
                if cont == 'n' or cont == 'q':
                    break
            except KeyboardInterrupt:
                print("\nQuitting...")
                break

        self.save_agent()
        self.print_stats()
        print("\nThanks for playing!")

    def train_self_play(self, n_games=100, temperature=None):
        """
        Train the agent by playing against itself.

        Args:
            n_games: Number of games to play
            temperature: Temperature for action selection (None = adaptive)
        """
        print(f"\n{'='*60}")
        print(f"Training: Agent playing against itself for {n_games} games")
        print("="*60)

        for game_num in range(1, n_games + 1):
            self._play_self_game(temperature=temperature)

            if game_num % 10 == 0:
                print(f"\nCompleted {game_num}/{n_games} training games")
                print(f"Total training steps: {self.agent.global_step}")
                self._print_training_stats()

                # Show trend if available
                if len(self.training_history) > 0:
                    recent = self.training_history[-1]
                    print(f"\nCurrent win rates:")
                    print(
                        f"  X: {recent['x_win_rate']:.1%}, O: {recent['o_win_rate']:.1%}, Draws: {recent['draw_rate']:.1%}")

        print(f"\n{'='*60}")
        print("Training complete!")
        print("="*60)
        self._print_training_stats()

        # Show overall trend
        if len(self.training_history) >= 2:
            print(f"\n{'='*60}")
            print("Learning Progress:")
            print("="*60)
            # Show last 5 checkpoints
            for i, hist in enumerate(self.training_history[-5:]):
                print(f"  Games {hist['games']-100}-{hist['games']}: "
                      f"X={hist['x_win_rate']:.1%}, O={hist['o_win_rate']:.1%}, "
                      f"Draws={hist['draw_rate']:.1%}")

        self.save_agent()
