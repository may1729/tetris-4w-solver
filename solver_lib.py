### IMPORTS ###

from collections import deque
import pickle
import sys

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

# Obtains all queues (ignoring bag structure) of length n
def all_queues(queue_length):
  if queue_length <= 0:
    yield ""
  else:
    for piece in PIECES:
      for queue in all_queues(queue_length-1):
        yield queue + piece

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

# Score a mino count
# This is based on distance from 6, 9, 12, 15 minos
def score_num_minos(mino_count):
  target = [12, 9, 6, 15][mino_count % 4]
  return abs(mino_count - target)

# Obtains queues that with the given hold could place the specified pieces
def get_input_queues_for_output_sequence(target, hold):
  if len(target) == 0:
    yield ""
  elif target[0] == hold:
    for piece in PIECES:
      for queue in get_input_queues_for_output_sequence(target[1:], piece):
        yield piece + queue
  else:
    for queue in get_input_queues_for_output_sequence(target[1:], hold):
      yield target[0] + queue

### MAIN FUNCTIONS ###

# Computes all possible piece placements given board and piece.
# Returns a list of all possible boards, with their finesse
# If no_breaks is True, then only returns placements that do not break combo
# Assume 100g.
# board_hash is the hash of the input board.
# piece is the next piece.
def get_next_boards(board_hash, piece, no_breaks = False, can180 = True):
  
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
  # current state -> (previous state, keypress list)
  previous = {}
  previous[(y, 1, 0)] = (None, ())
  
  # BFS on all possible ending locations for piece, assuming 100g
  while len(queue) > 0:
    current = queue.popleft()
    if current not in visited:
      visited.add(current)
      (y, x, rotation) = current
      
      # test movement
      for (x_move, x_finesse) in ((-1, "moveLeft"), (1, "moveRight")):
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
          newState = (y + new_y_offset, x + x_move, rotation)
          queue.append(newState)
          if newState not in previous:
            if new_y_offset < 0:
              previous[newState] = (current, (x_finesse, "softDrop"))
            else:
              previous[newState] = (current, (x_finesse,))
      
      vv = ((1, "rotateCW"), (2, "rotate180"), (3, "rotateCCW")) if can180 else ((1, "rotateCW"), (3, "rotateCCW"))
      # test rotation
      for (rotation_move, rotation_finesse) in vv:
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
            original_y_position = new_y_position
            while good:
              new_y_position -= 1
              for (offset_y, offset_x) in PIECES[piece][new_rotation]:
                (new_y, new_x) = (new_y_position + offset_y, new_x_position + offset_x)
                if (new_y, new_x) in square_set or not (0 <= new_y and 0 <= new_x < 4):
                  good = False
                  new_y_position += 1
                  break
            newState = (new_y_position, new_x_position, new_rotation)
            queue.append(newState)
            if newState not in previous:
              if original_y_position != new_y_position:
                previous[newState] = (current, (rotation_finesse, "softDrop"))
              else:
                previous[newState] = (current, (rotation_finesse,))
            break
  
  # Obtain board states
  # board_hash -> finesse
  boards = {}
  for (y, x, rotation) in visited:
    
    new_hash = board_hash
    for (offset_y, offset_x) in PIECES[piece][rotation]:
      new_hash += 2**(4 * (y + offset_y) + x + offset_x)
    new_board = unhash_board(new_hash)
    
    # Remove completed lines and check for breaks
    cleared_board = [_ for _ in new_board if 0 in _]
    kept_combo = len(cleared_board) != len(new_board)

    # Compute finesse
    if not no_breaks or kept_combo:
      finesse_groups = []
      current_state = (y, x, rotation)
      while current_state != None:
        (current_state, finesse_group) = previous[current_state]
        finesse_groups.append(finesse_group)
      finesse = []
      for finesse_group in reversed(finesse_groups):
        finesse += finesse_group
      
      cleared_board_hash = hash_board(cleared_board)
      if cleared_board_hash not in boards:
        boards[cleared_board_hash] = finesse
      elif len(finesse) < len(boards[cleared_board_hash]):
        boards[cleared_board_hash] = finesse
  
  return boards

# Computes all possible board states at the end of the given queue.
# board_hash is the hash of the input board.
# queue is a string containing the next pieces.
def get_next_boards_given_queue(board_hash, queue):
  boards = set([board_hash])
  for piece in queue:
    new_boards = set()
    for board in boards:
      new_boards = new_boards.union(set(get_next_boards(board, piece).keys()))
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

# saves transition_cache to pickle
def save_transition_cache(transition_cache, filename):
  output_file = open(filename, 'wb')
  pickle.dump(transition_cache, output_file)
  output_file.close()

# load transition_cache from pickle
def load_transition_cache(filename):
  input_file = open(filename, 'rb')
  transition_cache = pickle.load(input_file)
  input_file.close()
  return transition_cache

# # key: (board, piece); value: { [board]: finesse }
# def hash_states(d):
#   s = '';
  
#   for board, piece in d:
#     i = '';
#     for k in d[(board, piece)]:
#       i += str(k) + ":" + ",".join(d[(board, piece)][k]) + ";"
#     s += f'{board}&{piece}={i}/'
  
#   return s.strip()

# def unhash_states(s: str):
#   d = {}
#   for line in s.split("/"):
#     if line.strip() == "": continue
#     (key, value) = line.split("=")
#     (board, piece) = key.split("&")
#     i = {}
#     for ln in value.split(";"):
#       if ln.strip() == "": continue
#       (k,v) = ln.split(":")
#       i[k] = v.split(",")
#     d[(board, piece)] = i
  
#   return d
  