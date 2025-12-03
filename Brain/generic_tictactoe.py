import torch
import torch.nn as nn
import torch.optim as optim
import random

EMPTY = 0
X = 1
O = -1
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def check_winner(b):
    for a, b1, c in WIN_LINES:
        if b[a] != 0 and b[a] == b[b1] == b[c]:
            return b[a]
    if 0 not in b:
        return 0
    return None


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
        )

    def forward(self, x):
        return self.net(x)


def legal_mask(board):
    m = torch.tensor([1.0 if c == 0 else float("-inf") for c in board])
    return m


def play_episode(model):
    board = [0]*9
    player = 1
    logps = []
    while True:
        inp = torch.tensor(board, dtype=torch.float32)
        logits = model(inp)
        logits = logits + legal_mask(board)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logps.append((dist.log_prob(action), player))

        board[action.item()] = player
        w = check_winner(board)
        if w is not None:
            if w == 0:
                return logps, 0.0
            return logps, float(w)
        player = -player


def train(model, episodes=800, lr=0.01, entropy_coeff=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(episodes):
        logps, outcome = play_episode(model)
        opt.zero_grad()
        loss = 0
        for logp, player in logps:
            r = outcome * player
            loss -= logp * r
            loss -= entropy_coeff * (-logp.exp() * logp)  # entropy bonus
        loss.backward()
        opt.step()


def best_move(model, board):
    """Get the best move from the model for a given board state."""
    inp = torch.tensor(board, dtype=torch.float32)
    with torch.no_grad():
        logits = model(inp)
        logits = logits + legal_mask(board)
        return torch.argmax(logits).item()


def get_legal_moves(board):
    """Get list of legal move indices."""
    return [i for i in range(9) if board[i] == EMPTY]


def test_against_random(model, num_games=1000, model_player=X):
    """
    Test the model against a random player.

    Args:
        model: The trained PolicyNet model
        num_games: Number of games to play (default 1000)
        model_player: Which player the model plays as (X=1 or O=-1, default X)

    Returns:
        dict with statistics: wins, losses, draws, win_rate, loss_rate, draw_rate
    """
    model.eval()
    random_player = -model_player

    wins = 0
    losses = 0
    draws = 0

    for game in range(num_games):
        board = [EMPTY] * 9
        current_player = X  # X always goes first

        while True:
            if current_player == model_player:
                # Model's turn
                move = best_move(model, board)
            else:
                # Random player's turn
                legal = get_legal_moves(board)
                move = random.choice(legal)

            board[move] = current_player
            winner = check_winner(board)

            if winner is not None:
                if winner == 0:
                    draws += 1
                elif winner == model_player:
                    wins += 1
                else:
                    losses += 1
                break

            current_player = -current_player

    stats = {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_games': num_games,
        'win_rate': wins / num_games,
        'loss_rate': losses / num_games,
        'draw_rate': draws / num_games
    }

    return stats


def print_test_stats(stats):
    """Print test statistics in a readable format."""
    print("\n" + "="*50)
    print("Test Results Against Random Player")
    print("="*50)
    print(f"Total Games:     {stats['total_games']}")
    print(f"Wins:            {stats['wins']} ({stats['win_rate']*100:.2f}%)")
    print(
        f"Losses:          {stats['losses']} ({stats['loss_rate']*100:.2f}%)")
    print(f"Draws:           {stats['draws']} ({stats['draw_rate']*100:.2f}%)")
    print("="*50 + "\n")


# demo
if __name__ == "__main__":
    model = PolicyNet()
    print("Training model...")
    train(model, episodes=5000, lr=0.001, entropy_coeff=0.001)
    print("Training complete!")

    # Test against random player
    print("\nTesting against random player (1000 games)...")
    stats = test_against_random(model, num_games=1000, model_player=X)
    print_test_stats(stats)

    # deterministic self-play test
    print("Self-play demonstration:")
    board = [0]*9
    player = 1
    while True:
        a = best_move(model, board)
        board[a] = player
        w = check_winner(board)
        if w is not None:
            print(f"Final board: {board}, Winner: {w}")
            break
        player = -player
