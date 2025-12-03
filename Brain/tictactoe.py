import random

EMPTY = "."
MARKS = ["X", "O"]

# ---------- basic board utilities ----------

WIN_LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def check_winner(board):
    """Return 'X', 'O', 'draw', or None."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "draw"
    return None


# ---------- symmetries and canonical states ----------

# index maps: new_board[i] = old_board[MAP[i]]
IDX = list(range(9))

ROT90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]
ROT180 = [8, 7, 6, 5, 4, 3, 2, 1, 0]
ROT270 = [2, 5, 8, 1, 4, 7, 0, 3, 6]
REF_H = [2, 1, 0, 5, 4, 3, 8, 7, 6]
REF_V = [6, 7, 8, 3, 4, 5, 0, 1, 2]
REF_MAIN = [0, 3, 6, 1, 4, 7, 2, 5, 8]
REF_ANTI = [8, 5, 2, 7, 4, 1, 6, 3, 0]

SYMMETRIES = [
    IDX,
    ROT90,
    ROT180,
    ROT270,
    REF_H,
    REF_V,
    REF_MAIN,
    REF_ANTI,
]


def flip_marks(board):
    """Swap X and O, keep EMPTY."""
    res = []
    for c in board:
        if c == "X":
            res.append("O")
        elif c == "O":
            res.append("X")
        else:
            res.append(c)
    return "".join(res)


def canonical(board, mark_to_move):
    """
    Return (canonical_board, transform).

    canonical_board is the lexicographically smallest symmetric variant
    after possibly flipping marks so that the player to move is always 'X'.
    transform[i] gives the index in the original board that corresponds
    to canonical_board[i].
    """
    if mark_to_move == "X":
        base = board
    else:
        base = flip_marks(board)

    best = None
    best_map = None

    for M in SYMMETRIES:
        cand = "".join(base[M[i]] for i in range(9))
        if best is None or cand < best:
            best = cand
            best_map = M

    return best, best_map


# ---------- RL: Q table and policy ----------

class TicTacToeLearner:
    def __init__(self):
        # Q[state][action] = expected return for player to move
        self.Q = {}
        self.N = {}  # visit counts for step size 1/N

    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = [0.0] * 9
            self.N[state] = [0] * 9

    def choose_action(self, board, mark, epsilon):
        """Epsilon greedy from canonical state. Returns (real_action_index, state_key, action_in_canonical)."""
        canon, mapping = canonical(board, mark)
        self._ensure_state(canon)

        # find available actions in canonical board
        avail_can = [i for i, c in enumerate(canon) if c == EMPTY]

        if random.random() < epsilon:
            a_can = random.choice(avail_can)
        else:
            qvals = self.Q[canon]
            a_can = max(avail_can, key=lambda a: qvals[a])

        # map canonical action to real board index
        a_real = mapping[a_can]
        return a_real, canon, a_can

    def update_from_episode(self, episode, winner_id):
        """
        episode is list of (state_key, action_idx, player_id).
        winner_id is 0, 1, or None for draw.
        """
        for state, action, pid in episode:
            if winner_id is None:
                G = 0.0
            elif winner_id == pid:
                G = 1.0
            else:
                G = -1.0

            self._ensure_state(state)
            self.N[state][action] += 1
            n = self.N[state][action]
            alpha = 1.0 / n
            old = self.Q[state][action]
            self.Q[state][action] = old + alpha * (G - old)

    def play_self_game(self, epsilon):
        board = EMPTY * 9
        player_id = 0
        episode = []

        while True:
            mark = MARKS[player_id]
            move, state_key, a_can = self.choose_action(board, mark, epsilon)

            # apply move
            board = board[:move] + mark + board[move + 1:]

            episode.append((state_key, a_can, player_id))

            winner = check_winner(board)
            if winner is not None:
                if winner == "draw":
                    winner_id = None
                else:
                    winner_id = MARKS.index(winner)
                return episode, winner_id

            player_id = 1 - player_id

    def train_self_play(self, games=500, eps_start=0.3, eps_end=0.0):
        for g in range(games):
            # linear epsilon schedule
            if games > 1:
                epsilon = eps_start + (eps_end - eps_start) * (g / (games - 1))
            else:
                epsilon = eps_end

            episode, winner_id = self.play_self_game(epsilon)
            self.update_from_episode(episode, winner_id)

    # simple play-vs-agent helper
    def best_move(self, board, mark):
        """Deterministic greedy move from learned Q."""
        canon, mapping = canonical(board, mark)
        self._ensure_state(canon)
        avail_can = [i for i, c in enumerate(canon) if c == EMPTY]
        qvals = self.Q[canon]
        a_can = max(avail_can, key=lambda a: qvals[a])
        return mapping[a_can]


# ---------- demo ----------

if __name__ == "__main__":
    random.seed(0)

    agent = TicTacToeLearner()
    agent.train_self_play(games=500, eps_start=0.3, eps_end=0.0)

    # quick sanity check: have the agent play both sides once
    def play_det_game(agent):
        board = EMPTY * 9
        player_id = 0
        while True:
            mark = MARKS[player_id]
            move = agent.best_move(board, mark)
            board = board[:move] + mark + board[move + 1:]
            w = check_winner(board)
            if w is not None:
                return board, w
            player_id = 1 - player_id

    final_board, w = play_det_game(agent)
    print("Final board:", final_board[0:3], final_board[3:6], final_board[6:9])
    print("Result:", w)
