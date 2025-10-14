import lib.board_lib as board_lib

from collections import defaultdict, deque
import os

def generate_all_pc_queues(filename: str, num_pieces: int = 8, max_height: int = 8, override_existing_file: bool = False) -> dict:
  """Generate all PC queues with their intermediate states.

  Each line consists of the queue, followed by its intermediate board hashes, followed by the finesse list. For example:

  `TLSZ|hash,hash,hash|cw,l;ab,c;de,finish the rest of this later lol`

  Arguments:
    `filename` is the name of the file to output to, or read from.
    `num_pieces` is the maximum PC queue length to search for.
    `max_height` is the tallest allowable height of an intermediate state.
    `override_existing_file` if True will generate a new file even if one already exists.
  """
  # (queue, board_list) -> finesse
  pcs = {}

  if not override_existing_file and os.path.isfile(filename):
    with open(filename, 'r') as input_file:
      N = int(input_file.readline().strip())
      pcs = {}
      for _ in range(N):
        (pc_queue, board_hashes, finesse_list) = input_file.readline().strip().split("|")
        board_hashes = map(int, board_hashes.split(","))
        finesse_list = [piece_finesse.split(",") for piece_finesse in finesse_list.split(";")]
        pcs[(pc_queue, board_hashes)] = finesse_list
    return pcs
  
  max_height = min(num_pieces, max_height)  # max height
  max_board = 2**(4*max_height) - 1  # max hash
  
  # Optimization: use BFS forwards and backwards
  n_backwards = max(min(num_pieces - 2, 2), num_pieces//4 + 1)
  n_forwards = num_pieces - n_backwards
  
  # Backwards direction
  backwards_queue = deque()
  inital_state = (0, ("", (), ()))  # (board_hash, history)
  backwards_queue.append(inital_state)
  backwards_reachable_states = defaultdict(dict)  # board_hash -> {(queue, board_state_list): finesse}
  backwards_saved_transitions = {}  # (board_hash, piece) -> previous_board_list
  forwards_saved_transitions = {}  # (board_hash, piece) -> next_board_list
  
  visited = set()
  visited.add(inital_state)

  while len(backwards_queue) > 0:
    current = backwards_queue.popleft()
    (board_hash, history) = current
    (history_queue, history_boards, history_finesse) = history

    # Check each possible next piece
    for piece in board_lib.PIECES:
      new_history_queue = piece + history_queue
      if (board_hash, piece) not in backwards_saved_transitions:
        backwards_saved_transitions[(board_hash, piece)] = board_lib.get_previous_boards(board_hash, piece, forwards_saved_transitions)
      previous_boards = backwards_saved_transitions[(board_hash, piece)]
      # Track reachable board states
      for previous_board in previous_boards:
        (_, finesse) = previous_boards[previous_board]
        new_history_boards = (previous_board, *history_boards)
        new_history_finesse = (finesse, *history_finesse)
        new_history = (new_history_queue, new_history_boards, new_history_finesse)
        if previous_board != 0 and previous_board < max_board:
          backwards_reachable_states[previous_board][(new_history_queue, new_history_boards)] = new_history_finesse
          if len(new_history_queue) < n_backwards:
            next_state = (previous_board, new_history)
            if next_state not in visited:
              visited.add(next_state)
              backwards_queue.append(next_state)
  
  # Forwards direction
  forwards_queue = deque()
  inital_state = (0, ("", (), ()))  # (board_hash, history)
  forwards_queue.append(inital_state)
  # Dictionary for each board hash all the queues that produce it.
  forwards_reachable_states = defaultdict(dict)  # board_hash -> {(queue, board_state_list): finesse}
  
  visited = set()
  visited.add(inital_state)

  while len(forwards_queue) > 0:
    current = forwards_queue.popleft()
    (board_hash, history) = current
    (history_queue, history_boards, history_finesse) = history
    
    # Check each possible next piece
    for piece in board_lib.PIECES:
      new_history_queue = history_queue + piece
      if (board_hash, piece) not in forwards_saved_transitions:
        forwards_saved_transitions[(board_hash, piece)] = board_lib.get_next_boards(board_hash, piece)
      next_boards = forwards_saved_transitions[(board_hash, piece)]
      # Track reachable board states
      for next_board in next_boards:
        (_, finesse) = next_boards[next_board]
        new_history_boards = (*history_boards, next_board)
        new_history_finesse = (*history_finesse, finesse)
        new_history = (new_history_queue, new_history_boards, new_history_finesse)
        if next_board < max_board and next_board != 0:
          if next_board in backwards_reachable_states:
            forwards_reachable_states[next_board][(new_history_queue, new_history_boards)] = new_history_finesse
          if len(new_history_queue) < n_forwards:
            next_state = (next_board, new_history)
            if next_state not in visited:
              visited.add(next_state)
              forwards_queue.append(next_state)
  
  # Merge forwards and backwards
  for board_hash in forwards_reachable_states:
    if board_hash in backwards_reachable_states:
      for first_half in forwards_reachable_states[board_hash]:
        for second_half in backwards_reachable_states[board_hash]:
          combined_queue = first_half[0] + second_half[0]
          combined_board_states = first_half[1][:-1] + second_half[1]
          combined_finesse = forwards_reachable_states[board_hash][first_half] + backwards_reachable_states[board_hash][second_half]
          pcs[(combined_queue,combined_board_states)] = combined_finesse
  
  pcs[("I", ())] = [(),]  # Edge case
  
  # Save to output file
  with open(filename, 'w') as output_file:
    pc_list = sorted(pcs.keys(), key = lambda pc: (len(pc[0]), pc[0], pc[1]))
    output_file.write(str(len(pc_list)) + "\n")
    for (pc_queue, pc_hashes) in pc_list:
      finesse_list = pcs[(pc_queue, pc_hashes)]
      finesse_string = ";".join(",".join(piece_finesse) for piece_finesse in finesse_list)
      output_file.write(f"{pc_queue}|{','.join(str(pc_hash) for pc_hash in pc_hashes)}|{finesse_string}\n")
  return pcs

def get_pc_saves(piece_queue: str, pcs: set[str]) -> dict[str, str]:
  """Determines the set of saves for a given piece queue, given set of pcs.

  Returns a dictionary: `{saved_piece : pc_queue}`.
  If `saved_piece` is `?`, then there is no saved piece at the end: all pieces were used.

  `piece_queue` is a string containing the next pieces.

  `pcs` is the set of all pc queues to consider.
  """
  saves = {}
  for queue_order in board_lib.get_queue_orders(piece_queue):
    if queue_order[:-1] in pcs:
      saves[queue_order[-1]] = queue_order[:-1]
    if queue_order in pcs:
      saves["?"] = queue_order
  return saves

def max_pcs_in_queue(piece_queue: str) -> tuple[int, list[str]]:
  """Computes the maximum number of pcs that can be obtained in the given queue.
  """
  pcs = set(generate_all_pc_queues(board_lib.PC_QUEUES_FILENAME))  # set of all pcs
  max_n = len(max(pcs, key = lambda _:len(_)))  # length of longest pc
  piece_queue = piece_queue + "?"  # terminator character
  most_pcs_at_state = {(1, piece_queue[0]): (0, None, None)}  # (index, hold piece) -> (num pcs, previous state, previous solve)
  for index in range(1, len(piece_queue)):
    for hold in board_lib.PIECES:
      current_state = (index, hold)
      if current_state in most_pcs_at_state:
        for pieces_used in range(1, min(len(piece_queue) + 1 - index, max_n + 1)):
          pc_queue = hold + piece_queue[index:index + pieces_used]
          saves = get_pc_saves(pc_queue, pcs)
          for save in saves:
            next_state = (index + pieces_used, save)
            if next_state not in most_pcs_at_state or most_pcs_at_state[current_state][0] + 1 > most_pcs_at_state[next_state][0]:
              most_pcs_at_state[next_state] = (most_pcs_at_state[current_state][0] + 1, current_state, saves[save])
  (max_pcs, current_state, prev_solve) = max(most_pcs_at_state.values())
  if max_pcs == 0:
    return (0, [])
  reversed_history = [prev_solve,]
  while most_pcs_at_state[current_state][2] is not None:
    reversed_history.append(most_pcs_at_state[current_state][2])
    current_state = most_pcs_at_state[current_state][1]
  history = list(reversed(reversed_history))
  return (max_pcs, history)