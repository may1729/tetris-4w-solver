# minacode :oyes:
import json
from lib.combo_lib import get_best_next_combo_state
from lib.board_lib import hash_board, save_transition_cache, load_transition_cache

tc = {}
TC_FILE = "data/tc"

try:
    tc = load_transition_cache(TC_FILE)
except:
    pass
# board queue hold vision foresight can180 upstack
def p_combo(line: str):
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

    flags = a[5];

        
    can180 = 'f' in flags
    should_upstack = 'u' in flags
    use_hold = 'h' in flags
        
    if hold == '' and use_hold:
        queue = queue[1] + queue[0] + queue[2:]
    # print(f'{board} {queue}')

    get_best_next_combo_state(board_hash, hold + queue, foresight, can180, use_hold, should_upstack, tc)
    # print(f'stored {o}')
    return;

def p_pc(line: str):
    pass

while True:
    line = input()
    if line == "ex":
        save_transition_cache(tc, TC_FILE)
        print("ok")
    elif line.startswith("ren "):
        p_combo(line[4:])
    elif line.startswith("pc "):
        p_pc(line[3:])
    else:
        pass
        
