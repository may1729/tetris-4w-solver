### IMPORTS ###

from collections import defaultdict, deque
import math
import os
import random

### UTILITY FUNCTIONS ###

# Rotations, used for debugging
ROTATIONS = ["N", "E", "S", "W"]
ROTATION_MOVES = ["0", "CW", "180", "CCW"]

# File names
PIECES_FILENAME = "data/pieces.txt"
KICKS_FILENAME = "data/kicks.txt"
PC_QUEUES_FILENAME = "data/pc-queues.txt"

# Reads piece data from txt file.
# Returns {piece: {orientation: [squares relative to center]}}
# Stored as (y, x)
def get_pieces(filename):
  pieces = {}
  ifil = open(filename, 'r')
  piece_list = ifil.readline().strip()
  for piece in piece_list:
    pieces[piece] = {}
    squares = []
    row1 = ifil.readline().strip()
    row0 = ifil.readline().strip()
    for i in range(4):
      if row0[i] != ".": squares.append((0, i-1))
      if row1[i] != ".": squares.append((1, i-1))
    for rotation in range(4):
      pieces[piece][rotation] = tuple(squares)
      squares = [(-x, y) for (y, x) in squares]
  ifil.close()
  return pieces

PIECES = get_pieces(PIECES_FILENAME)
# Manually creating PIECE_WIDTH for now
PIECE_WIDTH = {piece:3 for piece in PIECES}
PIECE_WIDTH["O"] = 2
PIECE_WIDTH["I"] = 4

# Reads kick data from txt file.
# Returns {piece: {orientation: {input: [offset order]}}}
def get_kicks(filename):
  kicks = {}
  ifil = open(filename, 'r')
  for _p in range(7):
    piece = ifil.readline().strip()
    kicks[piece] = {}
    for orientation in range(4):
      kicks[piece][orientation] = {}
      for rotation_input in range(1, 4):
        ifil.readline()
        offsets = ifil.readline().strip().split("; ")
        piece_kicks = [tuple(map(int, _.split(", "))) for _ in offsets]
        kicks[piece][orientation][rotation_input] = piece_kicks
  ifil.close()
  return kicks

KICKS = get_kicks(KICKS_FILENAME)

# Converts a board state to an integer.
# Treats board state like binary string.
# Bits are read top to bottom, and right to left within each row.
def hash_board(board):
  board_hash = 0
  for row in reversed(board):
    for square in reversed(row):
      board_hash *= 2
      board_hash += square
  return board_hash

# Converts an integer to a board state.
def unhash_board(board_hash):
  board = []
  while board_hash > 0:
    row_hash = board_hash % 16
    board_hash //= 16
    board.append([])
    for square_num in range(4):
      board[-1].append(row_hash % 2)
      row_hash //= 2
  return board

# Computes number of minos in the board state corresponding to a hash.
def num_minos(board_hash):
  minos = 0
  while board_hash > 0:
    minos += board_hash % 2
    board_hash //= 2
  return minos

# Obtains list of squares in the board.
def get_square_list(board):
  square_list = []
  for y in range(len(board)):
    for x in range(4):
      if board[y][x]:
        square_list.append((y, x))
  return square_list

# Obtains list of ways to insert at most max_lines lines into a board
def lines_to_insert(board_height, max_lines):
  if max_lines == 1:
    yield ()
    for height in range(board_height+1):
      yield (height,)
  else:
    for line_set in lines_to_insert(board_height, max_lines - 1):
      yield (*line_set, board_height)
    if board_height > 0:
      for line_set in lines_to_insert(board_height - 1, max_lines):
        yield line_set
    else:
      yield ()

# Obtains all possible ways to play a queue given one hold.
def get_queue_orders(queue):
  if len(queue) == 1:
    yield queue[0]
    return
  for queue_order in get_queue_orders(queue[1:]):
    yield queue[0] + queue_order
  for queue_order in get_queue_orders(queue[0] + queue[2:]):
    yield queue[1] + queue_order

# Displays a board
def display_board(board_hash):
  board = unhash_board(board_hash)
  print("|    |")
  for row in reversed(board):
    print(f"|{''.join([[' ', '#'][_] for _ in row])}|")
  print("+----+")

# Displays a list of boards
def display_boards(board_hash_list):
  for board_hash in board_hash_list:
    display_board(board_hash)
    print()

### MAIN FUNCTIONS ###

# Computes all possible piece placements given board and piece.
# Returns a list of all possible boards.
# Assume 100g.
# board_hash is the hash of the input board.
# piece is the next piece.
def get_next_boards(board_hash, piece):
  
  # Obtain board
  board = unhash_board(board_hash)
  square_set = set(get_square_list(board))
  
  # Detect starting position of piece, assuming 100g
  y = len(board) - 1
  while True:
    if y < 0:
      y = 0
      break
    good = True
    for (offset_y, offset_x) in PIECES[piece][0]:
      if (y + offset_y, 1 + offset_x) in square_set:
        good = False
        break
    if not good:
      y += 1
      break
    y -= 1
  
  # State is (y, x, rotation)
  queue = deque()
  queue.append((y, 1, 0))
  visited = set()
  
  # BFS on all possible ending locations for piece, assuming 100g
  while len(queue) > 0:
    current = queue.popleft()
    if current not in visited:
      visited.add(current)
      (y, x, rotation) = current
      
      # test movement
      for x_move in (-1, 1):
        new_y_offset = 1
        good = True
        while good:
          new_y_offset -= 1
          for (offset_y, offset_x) in PIECES[piece][rotation]:
            (new_y, new_x) = (y + offset_y + new_y_offset, x + offset_x + x_move)
            if (new_y, new_x) in square_set or not (0 <= new_y and 0 <= new_x < 4):
              good = False
              new_y_offset += 1
              break
        if new_y_offset <= 0:
          queue.append((y + new_y_offset, x + x_move, rotation))
      
      # test rotation
      for rotation_move in (1, 2, 3):
        new_rotation = (rotation + rotation_move) % 4
        for (kick_offset_y, kick_offset_x) in KICKS[piece][rotation][rotation_move]:
          new_y_position = kick_offset_y + y
          new_x_position = kick_offset_x + x
          good = True
          for (offset_y, offset_x) in PIECES[piece][new_rotation]:
            (new_y, new_x) = (new_y_position + offset_y, new_x_position + offset_x)
            if (new_y, new_x) in square_set or not (0 <= new_y and 0 <= new_x < 4):
              good = False
              break
          if good:
            # gravity
            while good:
              new_y_position -= 1
              for (offset_y, offset_x) in PIECES[piece][new_rotation]:
                (new_y, new_x) = (new_y_position + offset_y, new_x_position + offset_x)
                if (new_y, new_x) in square_set or not (0 <= new_y and 0 <= new_x < 4):
                  good = False
                  new_y_position += 1
                  break
            queue.append((new_y_position, new_x_position, new_rotation))
            break
  
  # Obtain board states
  boards = set()
  for (y, x, rotation) in visited:
    new_hash = board_hash
    for (offset_y, offset_x) in PIECES[piece][rotation]:
      new_hash += 2**(4 * (y + offset_y) + x + offset_x)
    new_board = unhash_board(new_hash)
    
    # Remove completed lines
    cleared_board = [_ for _ in new_board if 0 in _]
    
    boards.add(hash_board(cleared_board))
  
  return sorted(boards)

# Computes all possible board states at the end of the given queue.
# board_hash is the hash of the input board.
# queue is a string containing the next pieces.
def get_next_boards_given_queue(board_hash, queue):
  boards = set([board_hash])
  for piece in queue:
    new_boards = set()
    for board in boards:
      new_boards = new_boards.union(set(get_next_boards(board, piece)))
    boards = new_boards
  return sorted(boards)

# Computes all possible previous piece placements given board and previous piece
# Returns a list of all possible previous boards.
# Assume 100g.
# board_hash is the hash of the input board.
# piece is the next piece.
# forwards_saved_transitions is a cache of the transitions of board states in the forwards direction.
def get_previous_boards(board_hash, piece, forwards_saved_transitions = {}):
  
  # Obtain board
  board = unhash_board(board_hash)
  
  # Obtain potential board states such that adding the given piece would result in current board state
  candidate_previous_boards = set()
  for line_list in lines_to_insert(len(board), PIECE_WIDTH[piece]):
    candidate_previous_board = []
    previous_index = 0
    for line_index in line_list:
      for row in range(previous_index, line_index):
        candidate_previous_board.append(board[row])
      candidate_previous_board.append([1, 1, 1, 1])
      previous_index = line_index
    for row in range(previous_index, len(board)):
      candidate_previous_board.append(board[row])
    
    # Look for positions where the given piece fits
    candidate_previous_board_hash = hash_board(candidate_previous_board)
    square_set = set(get_square_list(candidate_previous_board))
    for y in range(len(candidate_previous_board)):
      for x in range(4):
        for rotation in range(4):
          good = True
          piece_hash = 0
          for (offset_y, offset_x) in PIECES[piece][rotation]:
            (new_y, new_x) = (y + offset_y, x + offset_x)
            if (new_y, new_x) not in square_set:
              good = False
              break
            piece_hash += 2**(4 * new_y + new_x)
          
          # Compute hash and check for lack of filled in lines
          if good:
            processed_previous_board_hash = candidate_previous_board_hash - piece_hash
            processed_previous_board = unhash_board(processed_previous_board_hash)
            if False not in [0 in row for row in processed_previous_board]:
              candidate_previous_boards.add(processed_previous_board_hash)
  
  candidate_previous_boards = sorted(candidate_previous_boards)
  # Ensure it is possible to reach current board state from each candidate previous board state
  boards = []
  for candidate_previous_board in candidate_previous_boards:
    if (candidate_previous_board, piece) not in forwards_saved_transitions:
      forwards_saved_transitions[(candidate_previous_board, piece)] = get_next_boards(candidate_previous_board, piece)
    if board_hash in forwards_saved_transitions[(candidate_previous_board, piece)]:
      boards.append(candidate_previous_board)
  
  return boards

# Computes all possible board states that at the end of the given queue results in the given board.
# board_hash is the hash of the input board.
# queue is a string containing the next pieces.
def get_previous_boards_given_queue(board_hash, queue):
  boards = set([board_hash])
  forwards_saved_transitions = {}
  for piece in reversed(queue):
    prev_boards = set()
    for board in boards:
      prev_boards = prev_boards.union(set(get_previous_boards(board, piece, forwards_saved_transitions)))
    boards = prev_boards
  return sorted(boards)

# Computes best next board state for board state sequence
# that results in the fewest non line clears
# if len(queue) - 1 pieces are placed
# Among those, the one with the fewest minos
# Returns the piece used and the next board hash
def get_best_next_combo_state(board_hash, queue, foresight = 1, transition_cache = {}):
  BREAKS_LIMIT = 2  # ignore anything with more than BREAKS_LIMIT breaks
  for max_breaks in range(BREAKS_LIMIT+1):
    # (hold, board_state) -> set of (hold, queue_index, ending_board_state)
    immediate_placements = {}
    # (hold, queue_index, board_state) -> set of (hold, starting_board_state)
    reversed_immediate_placements = {}
    # (hold, queue_index, board_state) -> combo breaks
    least_breaks = {}
    # Track what we have explored.
    states_considered = set()

    # Cache the next boards. (board_hash, piece) -> reachable boards
    saved_next_boards = transition_cache
    def _cached_get_next_boards(_board_hash, _piece):
      if (_board_hash, _piece) not in saved_next_boards:
        saved_next_boards[(_board_hash, _piece)] = get_next_boards(_board_hash, _piece)
      return saved_next_boards[(_board_hash, _piece)]

    # (hold, queue_index, board_state)
    continuation_queue = deque()
    continuation_queue.append((queue[0], 1, board_hash))
    least_breaks[(queue[0], 1, board_hash)] = 0

    # BFS to see all ending states
    while len(continuation_queue) > 0:
      current_state = continuation_queue.popleft()
      (hold, queue_index, current_board_hash) = current_state
      current_mino_count = num_minos(current_board_hash)
      current_num_breaks = least_breaks[current_state]

      # Test not using hold, then using hold piece
      for (next_used, next_hold) in ((queue[queue_index], hold), (hold, queue[queue_index])):
        for next_board_hash in _cached_get_next_boards(current_board_hash, next_used):
          next_state = (next_hold, queue_index + 1, next_board_hash)

          # Update break counts
          next_mino_count = num_minos(next_board_hash)
          next_num_breaks = current_num_breaks + (next_mino_count > current_mino_count)
          if next_state not in least_breaks or next_num_breaks < least_breaks[next_state]:
            least_breaks[next_state] = next_num_breaks
            # Make sure we only add possible origin placements if the number of breaks along the way is the same
            # We do this by resetting reversed_immediate_placements[next_state]
            if queue_index >= 2:
              reversed_immediate_placements[next_state] = reversed_immediate_placements[current_state]
          
          # Add next states to queue
          if least_breaks[next_state] <= max_breaks:
            # Initialize immediate_placements and reversed_immediate_placements
            (next_hold, next_queue_index, next_board_hash) = next_state
            if next_queue_index == 2:
              immediate_placements[(next_hold, next_board_hash)] = set()
              reversed_immediate_placements[next_state] = set(((next_hold, next_board_hash),))
            
            # Update reversed_immediate_placements
            if next_queue_index >= 3:
              if next_state not in reversed_immediate_placements:
                reversed_immediate_placements[next_state] = reversed_immediate_placements[current_state]
              elif next_num_breaks == least_breaks[next_state]:
                reversed_immediate_placements[next_state] = reversed_immediate_placements[next_state].union(reversed_immediate_placements[current_state])

            # Actually add next states to queue
            if next_state not in states_considered:
              if queue_index + 1 < len(queue):
                states_considered.add(next_state)
                continuation_queue.append(next_state)

    # Do foresight
    # (hold, queue_index, board_state) -> set of accommodated next pieces
    accommodated = {}
    if foresight > 0:
      for final_state in least_breaks:
        accommodated[final_state] = set()
        # todo! for every queue test queue
        # ignore hold
        # after determining what queues work
        # for each instance where the hold appears, can replace with any piece
        # do dp to figure out what queues work
        (hold, queue_index, final_hash) = final_state
        current_mino_count = num_minos(final_hash)

        # Consider hold piece
        board_hash_list = _cached_get_next_boards(final_hash, hold)
        for next_board_hash in board_hash_list:
          if num_minos(next_board_hash) <= current_mino_count:
            for piece in PIECES:
              accommodated[final_state].add(piece)
            break
        
        # Consider each possible next piece
        if len(accommodated[final_state]) == 0:
          for piece in PIECES:
            board_hash_list = _cached_get_next_boards(final_hash, piece)
            for next_board_hash in board_hash_list:
              if num_minos(next_board_hash) <= current_mino_count:
                accommodated[final_state].add(piece)
                break

    # fill in immediate_placements
    for ending_state in reversed_immediate_placements:
      queue_index = ending_state[1]
      if queue_index == len(queue):
        for immediate_placement in reversed_immediate_placements[ending_state]:
          immediate_placements[immediate_placement].add(ending_state)

    # compute scores and return answer. Lower scores are better.
    # Score is 1000 * 10**foresight * expected number of breaks
    # We add 1000 for each unaccommodated next combination
    # We also add number based on future mino count to score
    # based on distance from 9, 10, 11, 12 minos

    best_end_state = None
    best_score = len(queue) * 1000000 * 10**foresight
    for state in immediate_placements:
      score = best_score + 1

      if foresight == 0:
        for end_state in immediate_placements[state]:
          # num breaks portion
          temp_score = least_breaks[end_state] * 1000 * 10**foresight
          # mino count portion
          temp_score += max(0, abs(2*num_minos(end_state[2]) - 21) - 3)//2
          # update best end score
          score = min(temp_score, score)
      
      # handle foresight
      else:
        max_num_breaks = 0
        
        # best mino score of anything that accommodates a queue
        currently_accommodated = {}

        for max_num_breaks in range(max_breaks, max_breaks + 2):
          for end_state in immediate_placements[state]:
            if least_breaks[end_state] == max_num_breaks:
              mino_score = max(0, abs(2*num_minos(end_state[2]) - 21) - 3)//2
              #mino_score = max(0, abs(num_minos(end_state[2]) - 42))
              for accommodated_queue in accommodated[end_state]:
                if accommodated_queue not in currently_accommodated:
                  currently_accommodated[accommodated_queue] = mino_score
                elif mino_score < currently_accommodated[accommodated_queue]:
                  currently_accommodated[accommodated_queue] = mino_score
          if len(currently_accommodated) > 0:
            break
        
        # num breaks portion
        score = max_num_breaks * 1000 * 10**foresight
        # We add 1000 for each unaccommodated next piece
        score += (7**foresight - len(currently_accommodated)) * 1000
        # mino count portion
        score += (10*sum(currently_accommodated.values()))//max(1, len(currently_accommodated))
      
      if score < best_score:
        best_score = score
        best_end_state = state

    if best_end_state != None:
      print(f'Given {queue}, best is hold {best_end_state}, score {best_score}')
      display_board(best_end_state[1])
      return best_end_state
  
  # bot kinda screwed so just pick the last thing it was thinking of
  best_end_state = state
  print("FK", flush = True)
  print(f'Given {queue}, best is hold {best_end_state}, score {best_score}')
  display_board(best_end_state[1])
  return best_end_state

# Computes best combo continuation
# if len(queue) - lookahead pieces are placed
# using lookahead previews and foresight prediction
# if finish is true, will attempt to place an additional lookahead - 1 pieces
# (may have suspicious placements at the end)
def get_best_combo_continuation(board_hash, queue, lookahead = 6, foresight = 0, finish = True):
  combo = []
  current_hash = board_hash
  hold = queue[0]
  window = queue[1:lookahead+1]

  for decision_num in range(len(queue) - 1):
    # compute next state
    next_state = get_best_next_combo_state(current_hash, hold + window, foresight)
    (hold, current_hash) = next_state
    combo.append(next_state)

    # compute next window
    if decision_num < len(queue) - lookahead - 1:
      window = window[1:]+queue[decision_num+lookahead+1]
    elif not finish:
      # no need to finish the queue off
      break
    else:
      window = window[1:]
    
  return combo

# inf ds simulator
# simulation_length is number of pieces to simulate
def simulate_inf_ds(simulation_length = 1000, lookahead = 6, foresight = 0, well_height = 8):
  def _piece_list():
    pieces = list(PIECES.keys())
    index = len(pieces)
    while True:
      if index == len(pieces):
        random.shuffle(pieces)
        index = 0
      yield pieces[index]
      index += 1
  pieces = _piece_list()
  combo = []
  combos = []

  # initialize game state
  max_hash = 0
  current_hash = 0
  current_minos = 0
  hold = next(pieces)
  window = ""
  for _ in range(lookahead):
    window += next(pieces)
  current_combo = 0

  # precompute garbage wells
  well_multiplier = (16**well_height - 1)//15
  wells = [row_code * well_multiplier for row_code in [7, 11, 13, 14]]

  tc = {}

  for decision_num in range(simulation_length):
    # compute next state
    next_state = get_best_next_combo_state(current_hash, hold + window, foresight, transition_cache=tc)
    (hold, current_hash) = next_state
    combo.append(next_state)

    # compute next window
    window = window[1:] + next(pieces)

    # handle combo logic
    minos = num_minos(current_hash)
    if minos <= current_minos:
      current_combo += 1
      current_minos = minos
    else:
      combos.append(current_combo)
      current_combo = 0
      current_minos = minos + 3 * well_height

      # add garbage!!!
      current_hash = current_hash * (16**well_height) + wells[random.randint(0, 3)]
    max_hash = max(max_hash, current_hash)
    if max_hash > 16**32:
      print("DEAD")
      break
  combos.append(current_combo)
  print(combos)
  height = 0
  while max_hash > 0:
    max_hash //= 16
    height += 1
  print(height)
  return combo

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
    ifil = open(filename, 'r')
    N = int(ifil.readline().strip())
    if not add_board_states:
      pcs = [ifil.readline().strip() for _ in range(N)]
    else:
      pcs = {}
      for _ in range(N):
        line = ifil.readline().strip().split("|")
        pcs[line[0]] = list(map(int, line[1:]))
    ifil.close()
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
      for piece in PIECES:
        new_history = piece + history
        if (board_hash, piece) not in backwards_saved_transitions:
          backwards_saved_transitions[(board_hash, piece)] = get_previous_boards(board_hash, piece, forwards_saved_transitions)
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
      for piece in PIECES:
        new_history = history + piece
        if (board_hash, piece) not in forwards_saved_transitions:
          forwards_saved_transitions[(board_hash, piece)] = get_next_boards(board_hash, piece)
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
  ofil = open(filename, 'w')
  pcs = sorted(pcs, key = lambda pc: (len(pc), pc))
  ofil.write(str(len(pcs)) + "\n")
  ofil.write("\n".join(pcs))
  ofil.close()
  return pcs

# Determines the set of saves for a given pc queue ("X" if no save), given set of pcs.
# piece_queue is a string containing the next pieces.
# pcs is the set of all pc queues to consider.
def get_pc_saves(piece_queue, pcs):
  saves = {}
  for queue_order in get_queue_orders(piece_queue):
    if queue_order[:-1] in pcs:
      saves[queue_order[-1]] = queue_order[:-1]
    if queue_order in pcs:
      saves["X"] = queue_order
  return saves

# Computes the maximum number of pcs that can be obtained in a queue.
# piece_queue is a string containing the next pieces.
def max_pcs_in_queue(piece_queue):
  pcs = set(generate_all_pc_queues(PC_QUEUES_FILENAME))  # set of all pcs
  max_n = len(max(pcs, key = lambda _:len(_)))  # longest pc
  piece_queue = piece_queue + "X"  # terminator character
  dp = {(1, piece_queue[0]): (0, None, None)}  # (index, hold piece) -> (num pcs, previous state, previous solve)
  for index in range(1, len(piece_queue)):
    for hold in PIECES:
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