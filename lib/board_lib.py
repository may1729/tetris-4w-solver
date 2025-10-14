### IMPORTS ###

from collections import deque
import os.path
import pickle
import random
import typing

### UTILITY FUNCTIONS ###

# Rotations, used for debugging
ROTATIONS = ["N", "E", "S", "W"]
ROTATION_MOVES = ["0", "CW", "180", "CCW"]

# File names
PIECES_FILENAME = os.path.dirname(__file__) + "/../data/pieces.txt"
KICKS_FILENAME = os.path.dirname(__file__) + "/../data/kicks.txt"
CORNERS_FILENAME = os.path.dirname(__file__) + "/../data/corners.txt"
PC_QUEUES_FILENAME = os.path.dirname(__file__) + "/../data/pc-queues.txt"

# Types
type Coordinate = tuple[int, int]
type CoordinateList = list[Coordinate]
type CoordinateSet = set[Coordinate]
type Board = list[list[int]]

def get_pieces_from_file(filename: str) -> dict[str, dict[int, CoordinateList]]:
  """Reads piece data from a text file.

  Returns a dictionary: `{piece: {orientation: [minos relative to center]}}`.
  Coordinates are stored as `(y, x)`.
  Center of piece should always be on the second line second character.
  """
  pieces = {}
  with open(filename, 'r') as input_file:
    piece_list = input_file.readline().strip()
    for piece in piece_list:
      pieces[piece] = {}
      minos = []
      row1 = input_file.readline().strip()
      row0 = input_file.readline().strip()
      for i in range(4):
        if row0[i] != ".": minos.append((0, i-1))
        if row1[i] != ".": minos.append((1, i-1))
      for rotation in range(4):
        pieces[piece][rotation] = tuple(minos)
        minos = [(-x, y) for (y, x) in minos]
  return pieces

def get_piece_widths(pieces: dict[str, dict[int, CoordinateList]]) -> dict[str, int]:
  piece_widths = {}
  for piece in pieces:
    left = min(pieces[piece][0], key=lambda k:k[0])[0]
    right = max(pieces[piece][0], key=lambda k:k[0])[0]
    piece_widths[piece] = right - left
  return piece_widths

PIECES = get_pieces_from_file(PIECES_FILENAME)
PIECE_WIDTHS = get_piece_widths(PIECES)

def get_kicks_from_file(filename: str) -> dict[str, dict[int, dict[int, CoordinateList]]]:
  """Reads kick data from a text file.

  Returns a dictionary: `{piece: {orientation: {input: [offset order]}}}`
  """
  kicks = {}
  with open(filename, 'r') as input_file:
    for _p in range(7):
      piece = input_file.readline().strip()
      kicks[piece] = {}
      for orientation in range(4):
        kicks[piece][orientation] = {}
        for rotation_input in range(1, 4):
          input_file.readline()
          offsets = input_file.readline().strip().split("; ")
          piece_kicks = [tuple(map(int, _.split(", "))) for _ in offsets]
          kicks[piece][orientation][rotation_input] = piece_kicks
  return kicks

KICKS = get_kicks_from_file(KICKS_FILENAME)

# Returns {piece: [list of corner_list]}
def get_corners_from_file(filename: str) -> dict[str, list[CoordinateList]]:
  """Reads corner data from a text file.

  Returns a dictionary: `{piece: [list of corner_list]}`
  """
  corners = {}
  with open(filename, 'r') as input_file:
    num_pieces = int(input_file.readline())
    for piece_num in range(num_pieces):
      piece = input_file.readline().strip()
      corners[piece] = []
      for rotation_num in range(4):
        offsets = input_file.readline().strip().split("; ")
        orientation_corner_list = [tuple(map(int, _.split(", "))) for _ in offsets]
        corners[piece].append(orientation_corner_list)
  return corners

CORNERS = get_corners_from_file(CORNERS_FILENAME)

def hash_board(board: Board) -> int:
  """Converts a board state to an integer.

  Treats board state like binary string. Bits are read top to bottom, and right to left within each row.
  """
  board_hash = 0
  for row in reversed(board):
    for square in reversed(row):
      board_hash *= 2
      board_hash += square
  return board_hash

def unhash_board(board_hash: int) -> Board:
  """Converts an integer to a board state.

  Treats board state like binary string. Bits are read top to bottom, and right to left within each row.
  """
  board = []
  while board_hash > 0:
    row_hash = board_hash % 16
    board_hash //= 16
    board.append([])
    for square_num in range(4):
      board[-1].append(row_hash % 2)
      row_hash //= 2
  return board

def num_minos(board_hash: int) -> int:
  """Computes number of minos in the board state corresponding to a hash."""
  minos = 0
  while board_hash > 0:
    minos += board_hash % 2
    board_hash //= 2
  return minos

def get_mino_list(board: Board) -> CoordinateList:
  """Obtains list of minos in the board."""
  mino_list = []
  for y in range(len(board)):
    for x in range(4):
      if board[y][x]:
        mino_list.append((y, x))
  return mino_list

def lines_to_insert(board_height: int, max_lines: int) -> typing.Generator[tuple[int, ...]]:
  """Obtains list of ways to insert at most `max_lines` lines into a board."""
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

def generate_7bag():
    """Generates an infinite number of 7bag pieces."""
    pieces = list(PIECES.keys())
    index = len(pieces)
    while True:
      if index == len(pieces):
        random.shuffle(pieces)
        index = 0
      yield pieces[index]
      index += 1

def all_queues(queue_length: int) -> typing.Generator[str]:
  """Generates all queues (ignoring bag structure) of length `queue_length`"""
  if queue_length <= 0:
    yield ""
  else:
    for piece in PIECES:
      for queue in all_queues(queue_length-1):
        yield queue + piece

def get_queue_orders(queue: str) -> typing.Generator[str]:
  """Obtains all possible ways to play a queue given one hold."""
  if len(queue) == 1:
    yield queue[0]
    return
  for queue_order in get_queue_orders(queue[1:]):
    yield queue[0] + queue_order
  for queue_order in get_queue_orders(queue[0] + queue[2:]):
    yield queue[1] + queue_order

def is_piece_location_valid(mino_set: CoordinateSet, mino_offsets: CoordinateList, piece_y: int, piece_x: int):
  """Determines if a piece location is in board bounds and does not intersect board."""
  for (offset_y, offset_x) in mino_offsets:
    y = piece_y + offset_y
    x = piece_x + offset_x
    if (y, x) in mino_set or y < 0 or x < 0 or x >= 4:
      return False
  return True

def is_piece_location_spin(mino_set: CoordinateSet, piece: str, rotation: int, piece_y: int, piece_x: int):
  """Determines if a piece location and rotation counts as a spin under 3 corner handheld rule.

  Assumes initial location is valid.
  Assumes previous action is a rotation.
  """
  if piece not in CORNERS:
    return False
  corner_count = 0
  for corner_num in range(4):
    corner_x = piece_x + CORNERS[piece][rotation][corner_num][0]
    corner_y = piece_y - CORNERS[piece][rotation][corner_num][1]
    if (
      corner_y < 0
      or not 0 <= corner_x < 4
      or (corner_y, corner_x) in mino_set
    ):
      corner_count += 1
  return corner_count >= 3

# Compute vertical position of piece after sonic drop, assuming initial location is valid
def get_gravity_y(mino_set: CoordinateSet, mino_offsets: CoordinateList, start_y: int, start_x: int):
  """Compute vertical position of piece after sonic drop.
  
  Assumes initial location is valid.
  """
  current_y = start_y - 1
  floor_y = -min([offset[0] for offset in mino_offsets])
  while current_y >= floor_y:
    is_floating = True
    for (offset_y, offset_x) in mino_offsets:
      if (current_y + offset_y, start_x + offset_x) in mino_set:
        is_floating = False
        break
    if not is_floating:
      break
    current_y -= 1
  current_y += 1
  return current_y

def display_board(board_hash: int) -> None:
  """Displays a board."""
  board = unhash_board(board_hash)
  print("|    |")
  for row in reversed(board):
    print(f"|{''.join([[' ', '#'][_] for _ in row])}|")
  print("+----+")

def display_boards(board_hash_list: list) -> None:
  """Displays a list of boards."""
  for board_hash in board_hash_list:
    display_board(board_hash)
    print()

def score_num_minos(mino_count: int) -> int:
  """Return mino score of a board with given mino count of `mino_count`.

  This is used for an additional tiebreak on assessing board quality.
  This is based on distance from 6, 9, 12, or 15 minos.
  This also strongly dislikes a mino count of 4.
  """
  if mino_count == 4:
    return 400
  target = [12, 9, 6, 15][mino_count % 4]
  return abs(mino_count - target)

def get_input_queues_for_output_sequence(target: str, hold: str) -> typing.Generator[str]:
  """Obtains queues that with the given hold could place the specified pieces."""
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

def get_next_boards(board_hash: int, piece: str, no_breaks: bool = False, can180: bool = True) -> dict[int, tuple[bool, list[str]]]:
  """Computes all possible piece placements given board and piece.

  Returns a list of all possible boards, with their finesse.

  `board_hash` is the hash of the input board.

  `piece` is the next piece.

  If `no_breaks` is True, then only returns placements that do not break combo.

  If `can180` is False, then excludes all 180 finesse.

  Assumes 100g.
  """
  # Ensure piece is valid
  if piece not in PIECES:
    return {}
  
  # Obtain board
  board = unhash_board(board_hash)
  mino_set = set(get_mino_list(board))
  
  # Detect starting position of piece, assuming 100g
  y = get_gravity_y(mino_set, PIECES[piece][0], len(board), 1)
  
  # State is (y, x, rotation)
  queue = deque()
  queue.append(((y, 1, 0), False))  # Base case
  # Set of states visited in queue
  visited = set()
  # (current state, is_spin) -> (previous state, keypress list)
  previous = {}
  previous[((y, 1, 0), False)] = (None, ())  # Base case
  
  # BFS on all possible ending locations for piece, assuming 100g
  while len(queue) > 0:
    current = queue.popleft()
    if current not in visited:
      visited.add(current)
      ((y, x, rotation), is_current_spin) = current
      
      # test movement
      valid_movement_list = ((-1, "moveLeft"), (1, "moveRight"))
      for (x_move, x_finesse) in valid_movement_list:
        new_y = y
        if is_piece_location_valid(mino_set, PIECES[piece][rotation], y, x + x_move):
          new_y = get_gravity_y(mino_set, PIECES[piece][rotation], y, x + x_move)
          newState = (new_y, x + x_move, rotation)
          queue.append((newState, False))
          if (newState, False) not in previous:
            if new_y != y:
              previous[(newState, False)] = (current, (x_finesse, "softDrop"))
            else:
              previous[(newState, False)] = (current, (x_finesse,))
      
      # test rotation
      valid_rotation_list = ((1, "rotateCW"), (2, "rotate180"), (3, "rotateCCW")) if can180 else ((1, "rotateCW"), (3, "rotateCCW"))
      for (rotation_move, rotation_finesse) in valid_rotation_list:
        new_rotation = (rotation + rotation_move) % 4
        for (kick_offset_y, kick_offset_x) in KICKS[piece][rotation][rotation_move]:
          rotated_y_position = kick_offset_y + y
          rotated_x_position = kick_offset_x + x
          if is_piece_location_valid(mino_set, PIECES[piece][new_rotation], rotated_y_position, rotated_x_position):
            # gravity
            new_y_position = get_gravity_y(mino_set, PIECES[piece][new_rotation], rotated_y_position, rotated_x_position)
            newState = (new_y_position, rotated_x_position, new_rotation)
            if rotated_y_position != new_y_position:
              queue.append((newState, False))
              if (newState, False) not in previous:
                previous[(newState, False)] = (current, (rotation_finesse, "softDrop"))
            else:
              # check for spins
              is_spin = is_piece_location_spin(mino_set, piece, new_rotation, new_y_position, rotated_x_position)
              queue.append((newState, is_spin))
              if (newState, is_spin) not in previous:
                previous[(newState, is_spin)] = (current, (rotation_finesse,))
            break
  
  # Obtain board states
  # board_hash -> (is_spin, finesse)
  boards = {}
  for ((y, x, rotation), is_spin) in visited:
    
    new_hash = board_hash
    for (offset_y, offset_x) in PIECES[piece][rotation]:
      new_hash += int(2**(4 * (y + offset_y) + x + offset_x))
    new_board = unhash_board(new_hash)

    # Remove completed lines and check for breaks
    cleared_board = [_ for _ in new_board if 0 in _]
    kept_combo = len(cleared_board) != len(new_board)

    # Compute finesse
    if not no_breaks or kept_combo:
      finesse_groups = []
      current_state = ((y, x, rotation), is_spin)
      while current_state is not None:
        (current_state, finesse_group) = previous[current_state]
        finesse_groups.append(finesse_group)
      finesse = []
      for finesse_group in reversed(finesse_groups):
        finesse += finesse_group
      
      cleared_board_hash = hash_board(cleared_board)
      if (
        cleared_board_hash not in boards
        or (is_spin and not boards[cleared_board_hash][0])
        or (is_spin == boards[cleared_board_hash][0] and len(finesse) < len(boards[cleared_board_hash][1]))
      ):
        boards[cleared_board_hash] = (is_spin, finesse)
  
  return boards

def get_next_boards_given_queue(board_hash: int, queue: str) -> list[int]:
  """Computes all possible board states at the end of the given queue.

  Returns a list of attainable board hashes.

  `board_hash` is the hash of the input board.

  `queue` is a string containing the next pieces.
  """
  boards = set([board_hash])
  for piece in queue:
    new_boards = set()
    for board in boards:
      new_boards = new_boards.union(set(get_next_boards(board, piece).keys()))
    boards = new_boards
  return sorted(boards)

def get_previous_boards(board_hash: int, piece: str, can180: bool = True, forwards_saved_transitions: dict | None = None) -> list[int]:
  """Computes all possible previous piece boards given board and piece.

  Returns a list of all possible previous boards such that placing the given piece somewhere in the board would result in the given board.

  `board_hash` is the hash of the input board.

  `piece` is the previous piece.

  If `can180` is False, then excludes all 180 finesse.

  `forwards_saved_transitions` is a cache containing the saved results of `get_next_boards`.

  Assumes 100g.
  """
  if forwards_saved_transitions is None:
    forwards_saved_transitions = {}
  
  # Obtain board
  board = unhash_board(board_hash)
  
  # Obtain potential board states such that adding the given piece would result in current board state
  candidate_previous_boards = set()
  for line_list in lines_to_insert(len(board), PIECE_WIDTHS[piece]):
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
    mino_set = set(get_mino_list(candidate_previous_board))
    for y in range(len(candidate_previous_board)):
      for x in range(4):
        for rotation in range(4):
          is_valid = True
          piece_hash = 0
          for (offset_y, offset_x) in PIECES[piece][rotation]:
            (new_y, new_x) = (y + offset_y, x + offset_x)
            if (new_y, new_x) not in mino_set:
              is_valid = False
              break
            piece_hash += 2**(4 * new_y + new_x)
          
          # Compute hash and check for lack of filled in lines
          if is_valid:
            processed_previous_board_hash = candidate_previous_board_hash - piece_hash
            processed_previous_board = unhash_board(processed_previous_board_hash)
            if False not in [0 in row for row in processed_previous_board]:
              candidate_previous_boards.add(processed_previous_board_hash)
  
  candidate_previous_boards = sorted(candidate_previous_boards)
  # Ensure it is possible to reach current board state from each candidate previous board state
  boards = []
  for candidate_previous_board in candidate_previous_boards:
    if (candidate_previous_board, piece) not in forwards_saved_transitions:
      forwards_saved_transitions[(candidate_previous_board, piece)] = get_next_boards(candidate_previous_board, piece, can180=can180)
    if board_hash in forwards_saved_transitions[(candidate_previous_board, piece)]:
      boards.append(candidate_previous_board)
  
  return boards

def get_previous_boards_given_queue(board_hash: int, queue: str) -> list[int]:
  """Computes all possible board states that at the end of the given queue results in the given board.

  Returns a list of starting board hashes.

  `board_hash` is the hash of the input board.

  `queue` is a string containing the previous pieces.
  """
  boards = set([board_hash])
  forwards_saved_transitions = {}
  for piece in reversed(queue):
    prev_boards = set()
    for board in boards:
      prev_boards = prev_boards.union(set(get_previous_boards(board, piece, forwards_saved_transitions)))
    boards = prev_boards
  return sorted(boards)

def save_transition_cache(transition_cache: dict, filename: str) -> None:
  """Saves `transition_cache` to `pickle`."""
  with open(filename, 'wb') as output_file:
    pickle.dump(transition_cache, output_file)

def load_transition_cache(filename: str) -> None:
  """Loads `transition_cache` from `pickle`."""
  with open(filename, 'rb') as input_file:
    transition_cache = pickle.load(input_file)
  return transition_cache