### IMPORTS ###

import libs.board_lib as board_lib

from collections import defaultdict, deque
import os

# Generate all PC queues for any possible queue
# Reads from output file if one exists.
# Otherwise, saves to output file because this is gonna take FOREVER.
# filename is the name of the file to output to, or read from.
# n is the length of the queue.
# h is the max height an intermediate board state can be.
# override, if true, will generate a new file even if one already exists.
# add_board_states, if true, will add the hashes of the board states after each queue.
def generate_all_pc_queues(filename, n = 8, h = 8, override = False, add_board_states = False):
  if not override and os.path.isfile(filename):
    with open(filename, 'r') as input_file:
      N = int(input_file.readline().strip())
      if not add_board_states:
        pcs = [input_file.readline().strip() for _ in range(N)]
      else:
        pcs = {}
        for _ in range(N):
          line = input_file.readline().strip().split("|")
          pcs[line[0]] = list(map(int, line[1:]))
    return pcs
  
  h = min(n, h)
  pcs = set()
  
  max_board = 2**(4*h) - 1  # max hash
  
  # Optimization: use BFS forwards and backwards
  n_backwards = n//4 + 1
  n_forwards = n - n_backwards
  
  # Backwards direction
  backwards_queue = deque()
  backwards_queue.append((0, ""))  # (board_hash, history)
  backwards_reachable_states = defaultdict(set)  # board_hash -> queue_set
  backwards_saved_transitions = {}  # (board_hash, piece) -> next_board_list
  forwards_saved_transitions = {}  # (board_hash, piece) -> next_board_list
  
  visited = set()
  while len(backwards_queue) > 0:
    current = backwards_queue.popleft()
    if current not in visited:
      visited.add(current)
      (board_hash, history) = current
      
      # Check each possible next piece
      for piece in board_lib.PIECES:
        new_history = piece + history
        if (board_hash, piece) not in backwards_saved_transitions:
          backwards_saved_transitions[(board_hash, piece)] = board_lib.get_previous_boards(board_hash, piece, forwards_saved_transitions)
        for previous_board in backwards_saved_transitions[(board_hash, piece)]:
          # Track reachable board states
          if previous_board != 0 and previous_board < max_board:
            backwards_reachable_states[previous_board].add(new_history)
            if len(new_history) < n_backwards:
              backwards_queue.append((previous_board, new_history))
  
  # Forwards direction
  forwards_queue = deque()
  forwards_queue.append((0, ""))  # (board_hash, history)
  forwards_reachable_states = defaultdict(set)  # board_hash -> queue_set
  
  visited = set()
  while len(forwards_queue) > 0:
    current = forwards_queue.popleft()
    if current not in visited:
      visited.add(current)
      (board_hash, history) = current
      
      # Check each possible next piece
      for piece in board_lib.PIECES:
        new_history = history + piece
        if (board_hash, piece) not in forwards_saved_transitions:
          forwards_saved_transitions[(board_hash, piece)] = board_lib.get_next_boards(board_hash, piece)
        for next_board in forwards_saved_transitions[(board_hash, piece)]:
          # Track reachable board states
          if next_board < max_board and next_board != 0:
            if next_board in backwards_reachable_states:
              forwards_reachable_states[next_board].add(new_history)
            if len(new_history) < n_forwards:
              forwards_queue.append((next_board, new_history))
  
  # Merge forwards and backwards
  for board_hash in forwards_reachable_states:
    if board_hash in backwards_reachable_states:
      for first_half in forwards_reachable_states[board_hash]:
        for second_half in backwards_reachable_states[board_hash]:
          pcs.add(first_half + second_half)
  
  pcs.add("I")  # Edge case
  
  # Save to output file
  with open(filename, 'w') as output_file:
    pcs = sorted(pcs, key = lambda pc: (len(pc), pc))
    output_file.write(str(len(pcs)) + "\n")
    output_file.write("\n".join(pcs))
  return pcs

# Determines the set of saves for a given pc queue ("X" if no save), given set of pcs.
# piece_queue is a string containing the next pieces.
# pcs is the set of all pc queues to consider.
def get_pc_saves(piece_queue, pcs):
  saves = {}
  for queue_order in board_lib.get_queue_orders(piece_queue):
    if queue_order[:-1] in pcs:
      saves[queue_order[-1]] = queue_order[:-1]
    if queue_order in pcs:
      saves["X"] = queue_order
  return saves

# Computes the maximum number of pcs that can be obtained in a queue.
# piece_queue is a string containing the next pieces.
def max_pcs_in_queue(piece_queue):
  pcs = set(generate_all_pc_queues(board_lib.PC_QUEUES_FILENAME))  # set of all pcs
  max_n = len(max(pcs, key = lambda _:len(_)))  # longest pc
  piece_queue = piece_queue + "X"  # terminator character
  dp = {(1, piece_queue[0]): (0, None, None)}  # (index, hold piece) -> (num pcs, previous state, previous solve)
  for index in range(1, len(piece_queue)):
    for hold in board_lib.PIECES:
      current_state = (index, hold)
      if current_state in dp:
        for pieces_used in range(1, min(len(piece_queue) + 1 - index, max_n + 1)):
          pc_queue = hold + piece_queue[index:index + pieces_used]
          saves = get_pc_saves(pc_queue, pcs)
          for save in saves:
            next_state = (index + pieces_used, save)
            if next_state not in dp or dp[current_state][0] + 1 > dp[next_state][0]:
              dp[next_state] = (dp[current_state][0] + 1, current_state, saves[save])
  (max_pcs, current_state, prev_solve) = max(dp.values())
  if max_pcs == 0:
    return (0, [])
  reversed_history = [prev_solve,]
  while dp[current_state][2] != None:
    reversed_history.append(dp[current_state][2])
    current_state = dp[current_state][1]
  history = list(reversed(reversed_history))
  return (max_pcs, history)