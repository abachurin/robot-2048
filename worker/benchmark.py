"""
Standalone Python RL trainer — no MongoDB/S3.
Usage: python benchmark.py [N] [episodes]
Writes log to benchmark_nN.log
"""
import sys
import time
import random
import numpy as np
from typing import Tuple


# === Game Logic ===

def create_table():
    table = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    score = 0
                    line = (a, b, c, d)
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, False)
                        continue
                    line_1 = [v for v in line if v]
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)
                            line_1[i], line_1[i + 1] = x + 1, 0
                    line_2 = [v for v in line_1 if v]
                    line_2 = line_2 + [0] * (4 - len(line_2))
                    table[line] = (line_2, score, line != tuple(line_2))
    return table

TABLE = create_table()

def get_max_tile(row):
    return int(1 << np.max(row))

class Game:
    def __init__(self):
        self.row = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.numMoves = 0
        self.moves = []
        self.tiles = []
        self.new_tile()
        self.new_tile()
        self.tiles = []

    def __str__(self):
        return '\n'.join([''.join([str(1 << val if val else 0) + ' ' * (10 - len(str(1 << val)))
                                   for val in j]) for j in self.row]) \
               + f'\n score = {self.score}, moves = {self.numMoves}, reached {get_max_tile(self.row)}\n'

    @staticmethod
    def empty(row):
        return np.where(row == 0)

    def new_tile(self):
        i, j = self.empty(self.row)
        tile = 1 if random.randrange(10) else 2
        pos = random.randrange(len(i))
        self.row[i[pos], j[pos]] = tile
        self.tiles.append([i[pos], j[pos], tile])

    @staticmethod
    def empty_count(row):
        return 16 - np.count_nonzero(row)

    @staticmethod
    def adjacent_pair_count(row):
        return 24 - np.count_nonzero(row[:, :3] - row[:, 1:]) - np.count_nonzero(row[:3, :] - row[1:, :])

    def game_over(self, row):
        if self.empty_count(self.row):
            return False
        return not self.adjacent_pair_count(row)

    @staticmethod
    def _left(row, score):
        change = False
        new_row = row.copy()
        new_score = score
        for i in range(4):
            line, sc, change_line = TABLE[tuple(row[i])]
            if change_line:
                change = True
                new_score += sc
                new_row[i] = line
        return new_row, new_score, change

    def pre_move(self, row, score, direction):
        new_row = np.rot90(row, direction) if direction else row
        new_row, new_score, change = self._left(new_row, score)
        if direction:
            new_row = np.rot90(new_row, 4 - direction)
        return new_row, new_score, change


# === Feature Functions ===

def f_2(x):
    x_vert = ((x[:3, :] << 4) + x[1:, :]).ravel()
    x_hor = ((x[:, :3] << 4) + x[:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor])

def f_3(x):
    x_vert = ((x[:2, :] << 8) + (x[1:3, :] << 4) + x[2:, :]).ravel()
    x_hor = ((x[:, :2] << 8) + (x[:, 1:3] << 4) + x[:, 2:]).ravel()
    x_ex_00 = ((x[1:, :3] << 8) + (x[1:, 1:] << 4) + x[:3, 1:]).ravel()
    x_ex_01 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[1:, 1:]).ravel()
    x_ex_10 = ((x[:3, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_ex_11 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[:3, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_ex_00, x_ex_01, x_ex_10, x_ex_11])

def f_4(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq])

def f_5(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1:3, 1:3] << 16) + (x[:2, 1:3] << 12) + (x[1:3, :2] << 8) +
                (x[2:, 1:3] << 4) + x[1:3, 2:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle])

def f_6(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1:3, 1:3] << 16) + (x[:2, 1:3] << 12) + (x[1:3, :2] << 8) +
                (x[2:, 1:3] << 4) + x[1:3, 2:]).ravel()
    y = np.minimum(x, 13)
    x_vert_6 = (537824 * y[0:2, 0:3] + 38416 * y[1:3, 0:3] + 2744 * y[2:, 0:3] +
                196 * y[0:2, 1:] + 14 * y[1:3, 1:] + y[2:, 1:]).ravel()
    x_hor_6 = (537824 * y[0:3, 0:2] + 38416 * y[0:3, 1:3] + 2744 * y[0:3, 2:] +
               196 * y[1:, 0:2] + 14 * y[1:, 1:3] + y[1:, 2:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle, x_vert_6, x_hor_6])

FEATURE_FUNCTIONS = {2: f_2, 3: f_3, 4: f_4, 5: f_5, 6: f_6}
PAR_SHAPE = {
    2: (24, 16 ** 2),
    3: (52, 16 ** 3),
    4: (17, 16 ** 4),
    5: (21, 16 ** 5),
    6: (33, 0)
}
CUTOFF_FOR_6_F = 14


# === QAgent ===

class QAgent:
    def __init__(self, n=4, alpha=0.25, decay=0.9, step=1000, min_alpha=0.01):
        self.n = n
        self.alpha = alpha
        self.decay = decay
        self.step = step
        self.min_alpha = min_alpha
        self.next_decay = step
        self.num_feat, self.size_feat = PAR_SHAPE[n]
        self.features = FEATURE_FUNCTIONS[n]
        self.best_score = 0
        self.max_tile = 0

        if n == 6:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist() + \
                           (np.random.random((12, CUTOFF_FOR_6_F ** 6)) / 100).tolist()
        elif n == 5:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist()
        else:
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()

    def evaluate(self, row):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])

    def update(self, row, dw):
        for _ in range(4):
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.transpose(row)
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.rot90(np.transpose(row))

    def episode(self):
        game = Game()
        state, old_label = None, 0
        while not game.game_over(game.row):
            action, best_value = 0, -np.inf
            best_row, best_score = None, None
            for direction in range(4):
                new_row, new_score, change = game.pre_move(game.row, game.score, direction)
                if change:
                    value = self.evaluate(new_row)
                    if value > best_value:
                        action, best_value = direction, value
                        best_row, best_score = new_row, new_score
            if state is not None:
                dw = (best_score - game.score + best_value - old_label) * self.alpha / self.num_feat
                self.update(state, dw)
            game.row, game.score = best_row, best_score
            game.numMoves += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()
        game.moves.append(-1)
        dw = - old_label * self.alpha / self.num_feat
        self.update(state, dw)
        return game


# === Runner ===

def run(n, episodes, alpha=0.25):
    log_path = f'benchmark_n{n}.log'
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f'Python benchmark: N={n}, alpha={alpha}, episodes={episodes}')
    t0 = time.time()
    agent = QAgent(n=n, alpha=alpha)
    total_weights = sum(len(w) for w in agent.weights)
    log(f'Weights: {total_weights} entries')
    log(f'Init time: {time.time() - t0:.2f}s\n')

    start = time.time()
    start_1000 = start
    ma100 = []
    av1000 = []
    reached = [0] * 7
    best_of_1000 = Game()

    for ep in range(1, episodes + 1):
        game = agent.episode()
        ma100.append(game.score)
        av1000.append(game.score)
        max_tile = np.max(game.row)

        if game.score > best_of_1000.score:
            best_of_1000 = game
            if game.score > agent.best_score:
                agent.best_score = game.score
                log(f'\nNew best game at episode {ep}!\n{game.__str__()}')

        if max_tile >= 10:
            reached[min(max_tile - 10, 6)] += 1
        agent.max_tile = max(agent.max_tile, max_tile)

        if ep % 100 == 0:
            average = int(np.mean(ma100))
            log(f'episode {ep}, last 100 average = {average}')
            ma100 = []

        if ep % 1000 == 0:
            average = int(np.mean(av1000))
            elapsed_1000 = time.time() - start_1000
            log(f'\n=== Episode {ep} ===')
            log(f'{elapsed_1000:.1f}s for last {len(av1000)} episodes')
            log(f'average score = {average}')
            for j in range(7):
                r = round(sum(reached[j:]) / len(av1000) * 100, 2)
                if r:
                    log(f'{1 << (j + 10)} reached in {r}%')
            log(f'best game of last 1000:\n{best_of_1000.__str__()}')
            log(f'best game ever: score={agent.best_score}')
            av1000 = []
            reached = [0] * 7
            best_of_1000 = Game()
            start_1000 = time.time()

        if ep == agent.next_decay and agent.alpha > agent.min_alpha:
            agent.alpha = round(max(agent.alpha * agent.decay, agent.min_alpha), 6)
            agent.next_decay = ep + agent.step
            log(f'  LR decayed to {agent.alpha}')

    total = time.time() - start
    log(f'\nTotal time: {total:.1f}s ({episodes / total:.0f} episodes/sec)')

    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'\nLog written to {log_path}')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(n, episodes)
