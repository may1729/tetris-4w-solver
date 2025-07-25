# minacode :oyes:
import json
from combo_lib import get_best_next_combo_state
from solver_lib import hash_board, save_transition_cache, load_transition_cache

tc = {}
TC_FILE = "data/tc"

try:
    tc = load_transition_cache(TC_FILE)
except:
    pass
# board queue hold vision foresight
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
        
    can180 = bool(int(a[5]))
        
    # print(f'{board} {queue}')

    get_best_next_combo_state(board_hash, hold + queue, foresight, can180, tc)
    # print(f'stored {o}')
    return;

while True:
    line = input()
    if line == "ex":
        save_transition_cache(tc, TC_FILE)
        print("ok")
    else:
        f(line)
        
