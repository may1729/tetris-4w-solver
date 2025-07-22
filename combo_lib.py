### IMPORTS ###

import solver_lib

from collections import deque
import random

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
        saved_next_boards[(_board_hash, _piece)] = solver_lib.get_next_boards(_board_hash, _piece)
      return saved_next_boards[(_board_hash, _piece)]

    # (hold, queue_index, board_state)
    continuation_queue = deque()
    continuation_queue.append((queue[0], 1, board_hash))
    least_breaks[(queue[0], 1, board_hash)] = 0

    # BFS to see all ending states
    while len(continuation_queue) > 0:
      current_state = continuation_queue.popleft()
      (hold, queue_index, current_board_hash) = current_state
      current_mino_count = solver_lib.num_minos(current_board_hash)
      current_num_breaks = least_breaks[current_state]

      # Test not using hold, then using hold piece
      for (next_used, next_hold) in ((queue[queue_index], hold), (hold, queue[queue_index])):
        for next_board_hash in _cached_get_next_boards(current_board_hash, next_used):
          next_state = (next_hold, queue_index + 1, next_board_hash)

          # Update break counts
          next_mino_count = solver_lib.num_minos(next_board_hash)
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

    # fill in immediate_placements
    for ending_state in reversed_immediate_placements:
      queue_index = ending_state[1]
      if queue_index == len(queue):
        for immediate_placement in reversed_immediate_placements[ending_state]:
          immediate_placements[immediate_placement].add(ending_state)

    # Do foresight
    # (hold, queue_index, board_state) -> {set of accommodated next queues -> mino score}
    accommodated = {}
    if foresight > 0:
      for final_state in least_breaks:
        (hold, queue_index, final_hash) = final_state
        current_mino_count = solver_lib.num_minos(final_hash)
        accommodated[final_state] = {}
        # foresight queue -> best score
        foresight_scores = {}
        for foresight_queue in solver_lib.all_queues(foresight):
          foresight_scores[foresight_queue] = 1000

          # (foresight_queue_index, foresight_board_hash)
          foresight_continuation_queue = deque()
          foresight_continuation_queue.append((0, final_hash))

          # bfs on all ending states that do not break
          # no need to account for hold
          while len(foresight_continuation_queue) > 0:
            current_state = foresight_continuation_queue.popleft()
            (foresight_queue_index, foresight_board_hash) = current_state
            current_mino_count = solver_lib.num_minos(foresight_board_hash)

            for next_board_hash in _cached_get_next_boards(foresight_board_hash, foresight_queue[foresight_queue_index]):
              next_state = (foresight_queue_index + 1, next_board_hash)
              next_mino_count = solver_lib.num_minos(next_board_hash)
              if solver_lib.num_minos(next_board_hash) <= current_mino_count:
                if foresight_queue_index == foresight - 1:
                  # scoring
                  foresight_scores[foresight_queue] = min(foresight_scores[foresight_queue], solver_lib.score_num_minos(next_mino_count))
                else:
                  # add to bfs queue
                  foresight_continuation_queue.append(next_state)
        
        # compute accommodated scores
        for foresight_queue in foresight_scores:
          if foresight_scores[foresight_queue] != 1000:
            for input_queue in solver_lib.get_input_queues_for_output_sequence(foresight_queue, hold):
              if input_queue not in accommodated[final_state]:
                accommodated[final_state][input_queue] = foresight_scores[foresight_queue]
              elif foresight_scores[foresight_queue] < accommodated[final_state][input_queue]:
                accommodated[final_state][input_queue] = foresight_scores[foresight_queue]

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
          temp_score += solver_lib.score_num_minos(solver_lib.num_minos(end_state[2]))
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
              for accommodated_queue in accommodated[end_state]:
                if accommodated_queue not in currently_accommodated:
                  currently_accommodated[accommodated_queue] = accommodated[end_state][accommodated_queue]
                elif accommodated[end_state][accommodated_queue] < currently_accommodated[accommodated_queue]:
                  currently_accommodated[accommodated_queue] = accommodated[end_state][accommodated_queue]
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
      break
  
  if best_end_state == None:
    # bot kinda screwed so just pick the last thing it was thinking of
    best_end_state = state
    # print("FK", flush = True)
  
  (end_hold, end_hash) = best_end_state
  finesse_list = []
  used = queue[1]
  if end_hold != queue[0]:
    used = queue[0]
    finesse_list.append("hold")
  finesse_list += _cached_get_next_boards(board_hash, used)[end_hash]
  z=','.join(finesse_list)
  print(f'{used} {z}')
  # print(f'Finesse list: {finesse_list}')
  
  # h = solver_lib.hash_states(transition_cache);
  # t = solver_lib.unhash_states(h);
  # ile = open('data/transition_cache', 'w');
  # ile.write(h);
  # print(t)
  # solver_lib.display_board(board_hash)
  # solver_lib.display_board(end_hash)
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
    pieces = list(solver_lib.PIECES.keys())
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
    minos = solver_lib.num_minos(current_hash)
    if minos <= current_minos:
      current_combo += 1
      current_minos = minos
    else:
      combos.append(current_combo)
      current_combo = 0
      current_minos = minos + 3 * well_height

      # add garbage!!!
      current_hash = current_hash * (16**well_height) + wells[random.randint(0, 3)]
    
    # display board and game state
    solver_lib.display_board(current_hash)
    print(f"Combo: {current_combo}")

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