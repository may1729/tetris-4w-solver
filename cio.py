# minacode :oyes:

from combo_lib import get_best_next_combo_state
from solver_lib import hash_board, unhash_states

tc = {}
#
def f(line: str):
    global tc
    a = line.split(" ")
    # print(a)
    board = list(reversed([[1 if c == 'X' else 0 for c in row] for row in a[0].split('|')]))
    board_hash = hash_board(board)
    
    # print(board_hash)

    vision = int(a[3])
    queue = a[1][:vision]
    hold = '' if a[2] == '_' else a[2]
    foresight = int(a[4])

    if hold == '':
        queue = queue[1] + queue[0] + queue[2:]
        
    # print(f'{board} {queue}')

    o = get_best_next_combo_state(board_hash, hold + queue, foresight, tc)
    tc = o[1]
    # print(f'stored {o}')
    return;

while True:
    line = input()
    f(line)
        

