# minacode :oyes:

from solver_lib import get_best_next_combo_state, hash_board
import sys

a = sys.argv

# board, queue, hold, vision, foresight
board = list(reversed([[1 if c == 'X' else 0 for c in row] for row in a[1].split('|')]))
board_hash = hash_board(board)

vision = int(a[4])
queue = a[2][:vision]
hold = a[3]
foresight = int(a[5])

o = get_best_next_combo_state(board_hash, hold + queue, foresight)
print(o)