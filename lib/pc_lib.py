import lib.board_lib as board_lib
from lib.board_lib import BoardHash, Piece, PieceFinesse, Queue
from lib.board_lib import EMPTY_BOARD_HASH

from collections import defaultdict, deque
import os
from typing import Dict, List, Optional, Set, Tuple, TypeAlias

PC_State_Transitions: TypeAlias = Dict[Tuple[BoardHash, Piece], Dict[BoardHash, PieceFinesse]]

def generate_all_pc_queues(
    filename: str,
    num_pieces: int = 8,
    max_height: int = 8,
    override_existing_file: bool = False
  ) -> Dict[Tuple[Queue, Tuple[BoardHash]], List[PieceFinesse]]:
  """Generate all PC queues with their intermediate states.

  Each line consists of the queue, followed by its intermediate board hashes, followed by the finesse list. For example:

  `TLSZ|305,2097,201|cw;r,sd;f,cw;r,ccw,ccw`

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
        board_hashes = tuple(map(int, board_hashes.split(",")))
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

def build_state_transitions(
    pcs: Dict[Tuple[Queue, Tuple[BoardHash]], List[PieceFinesse]]
  ) -> PC_State_Transitions:
  """Builds board state transition graph given pc data.

  Returns, for each input board hash and input piece, all possible resulting board states with the finesse needed.
  """
  pc_state_transitions = {}
  for piece in board_lib.PIECES:
    pc_state_transitions[(EMPTY_BOARD_HASH, piece)] = {}

  for board_state in pcs:
    (queue, board_hash_list) = board_state
    finesse_list = pcs[board_state]
    board_hash_list.append(EMPTY_BOARD_HASH)
    n = len(queue)

    for piece_num in range(n):
      initial_state = (board_hash_list[piece_num - 1], queue[piece_num])
      if initial_state not in pc_state_transitions:
        pc_state_transitions[initial_state] = {}
      pc_state_transitions[initial_state][board_hash_list[piece_num]] = finesse_list[piece_num]
  
  return pc_state_transitions

def build_pc_distances(
    pcs: Dict[Tuple[Queue, Tuple[BoardHash]], List[PieceFinesse]]
  ) -> Dict[BoardHash, int]:
  """Computes, for each board hash, the minimum number of pieces to PC."""
  pc_distances = {}
  pc_distances[EMPTY_BOARD_HASH] = 1

  for (queue, board_hash_list) in pcs:
    n = len(queue)
    for piece_num in range(n - 1):
      board_hash = board_hash_list[piece_num]
      distance = n - 1 - piece_num
      if board_hash not in pc_distances or pc_distances[board_hash] > distance:
        pc_distances[board_hash] = distance
    
  return pc_distances

def compute_pieces_to_next_pc(transitions: PC_State_Transitions) -> Dict[Tuple[BoardHash, Piece], float]:
  """Computes expected number of pieces before next PC.
  
  Returns, for each pair of board hash and hold piece, the expected number of pieces before the next perfect clear.
  """
  # TODO: HELP HOW DO YOU DO THIS AAAAA
  # For each of 7 next pieces, Minimum of using hold piece and using given piece. Assume infinite vision.
  # For each board hash, look up its location in the tree and then gather all the prefixes to pc.
  # Then compute the average there, using the hold functions too.
  # This will take FOREVER though....
  return {}

def get_pc_saves(piece_queue: Queue, pcs: Set[Queue]) -> Dict[Piece, Queue]:
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

def max_pcs_in_queue(piece_queue: Queue) -> Tuple[int, List[Queue]]:
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

def get_best_next_pc_state(
    board_hash: BoardHash,
    queue: Queue,
    pc_set: Set[Queue],
    transitions: PC_State_Transitions,
    pc_distances: Dict[BoardHash, int],
    foresight: int = 1,
    canHold: bool = True
  ) -> Tuple[BoardHash, Piece, PieceFinesse]:
  """Computes best next board state.

  First, attempts to maximize expected number of pcs over next `len(queue) - 1 + foresight` pieces.
  Among those, minimizes the minimum distance from next pc.

  Returns a tuple with the next board hash, piece used, and finesse.

  `board_hash` is the hash of the input board.

  `queue` is a string containing the hold piece followed by the next pieces.

  `pc_set` is a set of all PC queues, the universe of possible PCs.

  `transitions` is the pc state transitions dictionary, the universe of possible moves.

  `pc_distances` maps board hashes to the minimum pieces required to PC.

  ~~If `canHold` is False, then assumes that the hold queue is empty and disallows access to it.~~
  """
  # (queue_index, board_hash, hold) -> set((board_hash, hold))
  first_placements = defaultdict(set)

  hold = queue[0]
  continuation_queues = defaultdict(set)
  # Initial state
  continuation_queues[1].add((board_hash, queue[0]))
  # Stores if a pc has been found
  can_pc = False

  # BFS to find moves that produce empty boards
  for queue_index in range(1, len(queue)):
    for current_state in continuation_queues[queue_index]:
      (current_board_hash, hold) = current_state
      # Test not using hold, then using hold piece
      for (next_used, next_hold) in ((queue[queue_index], hold), (hold, queue[queue_index])):
        for next_board_hash in transitions[(current_board_hash, next_used)]:
          next_state = (next_board_hash, next_hold)

          if queue_index == 1:
            # Initialize first_placements
            first_placements[(queue_index + 1, *next_state)] = set([next_state])
          else:
            # Update first_placements
            previous_first_placements = first_placements[(queue_index, *current_state)]
            first_placements[(queue_index + 1, *next_state)] = first_placements[(queue_index + 1, *next_state)].union(previous_first_placements)
          
          # Update continuation_queues
          if next_board_hash != EMPTY_BOARD_HASH:
            continuation_queues[queue_index + 1].add(next_state)
          else:
            can_pc = True

  # Construct set of states reachable from first placements
  # (board_hash, hold) -> set((queue_index, board_hash, hold))
  reachables = defaultdict(set)
  max_foresight_score = foresight + max(pc_distances.values()) + 3
  if not can_pc:
    num_foresight_queues = 7**foresight
    foresight_scores = {}
    for final_state in first_placements:
      if queue_index != len(queue):
        continue
      foresight_scores[final_state] = {}
      for first_state in first_placements[final_state]:
        reachables[first_state].add(final_state)
    
    for foresight_queue in board_lib.all_queues(foresight):
      # (foresight_queue_index, board_hash, hold) -> set(final_state)
      foresight_first_placements = defaultdict(set)

      foresight_continuation_queues = defaultdict(set)
      # Initial states
      for final_state in foresight_scores:
        foresight_continuation_queues[0].add(final_state)
        foresight_first_placements[final_state].add(final_state)
        foresight_scores[final_state] = max_foresight_score

      # BFS to find pcs
      # no need to account for hold
      # Maybe instead of doing bfs, do the big precompute
      # Store in prefix tree data structure for speed


    # For each foresight queue, compute pc score of each.
    # Pc score in this case is average number of pieces to pc.
    # If no pc, then do foresight + pc_distance

  # Then update immediate_placements depending on below:
  # If there are empty boards:

  # BFS until no more possible PCs

  # Then do foresight - For each foresight queue, max(For each suffix, number of pcs at the end)

  # Okay but what if there's no ending states. Do we just go :stare: again
  # Though there has to be no way to exit the universe right.
  pass