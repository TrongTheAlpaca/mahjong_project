from pathlib import Path
import json
import utilities.utilities as util
import pandas as pd


def flip_bit(bits, flip_index):
    return bits ^ (1 << 135 - flip_index)


def turn_on_bit(bits, on_index):
    return bits | (1 << 135 - on_index)


def turn_off_bit(bits, off_index):
    return bits & ~(1 << (135 - off_index))


INIT = 'Init'
DRAW = 'Draw'
DISCARD = 'Discard'
CHI = 'Chi'
PON = 'Pon'
MINKAN = 'MinKan'  # Open Kan
ANKAN = 'AnKan'  # Closed Kan
KAKAN = 'KaKan'  # Added Kan
NUKI = 'Nuki'  # Declare North dora
REMOVE = 'Remove'
REACH = 'Riichi'

# years = util.get_all_logs_annually(Path('D:') / 'logs', years=[2019], filterblade=False)
# all_logs = [Path('2014010100gm-00a9-0000-defe4a16.json')]

# game_data = pd.read_parquet(Path('E:') / 'mahjong' / 'pandas' / 'log_game_data.parquet',
#                             engine='fastparquet')  # Use `fastparquet` to preserve categorical data
# game_data = util.filter_logs(game_data)

# logs = util.get_logs(Path('D:') / 'logs', n_logs=50, years=[2019], progress_bar=True)

years = util.get_all_logs_annually(Path('E:') / 'mahjong' / 'logs',
                                   years=[2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
                                   progress_bar=True,
                                   filterblade=True)

for year, logs in years:

    output_path = Path('E:') / 'mahjong' / 'state_data' / str(year)
    output_path.mkdir(parents=True, exist_ok=True)

    for log_json in logs:

        rows = []  # List of Dictionaries

        log = json.load(log_json.open())

        for round_number, actions in enumerate(log['rounds']):

            # STEP 1: Handle INIT case in isolation from the rest
            # Rationale:
            #   A log always begin with INIT, isolate it
            #   such that next step can forgo an single-time-use if-statement

            init = actions.pop(0)

            init_data = init['data']

            rows.append({
                'log_id': log_json.stem,
                'round': round_number,
                'step': 0,
                'player': -1,
                'action': INIT,
                'tile': 255
            })

            for player, starting_hand in enumerate(init_data['hands']):
                for tile in starting_hand:
                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': 0,
                        'player': player,
                        'action': DRAW,
                        'tile': tile
                    })

            rows.append({
                'log_id': log_json.stem,
                'round': round_number,
                'step': 0,
                'player': -1,
                'action': 'Dora',
                'tile': init_data['dora']  # Tile revealed
            })

            # STEP 2: Handle actions after INIT
            # Loop Params
            pons = [set(), set(), set(), set()]  # Needed for Chankan = Robbing a Kan
            previous_kakan_tile = -1  # Needed for Chankan = Robbing a Kan
            discard_tile = -1  # Needed when removing from pool
            step_offset = 1
            next_discard_is_reach = False

            for step, action in enumerate(actions):

                tag = action['tag']

                if tag == 'DRAW':

                    previous_drawn_tile = action['data']['tile']

                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': step + step_offset,
                        'player': action['data']['player'],
                        'action': DRAW,
                        'tile': previous_drawn_tile
                    })

                elif tag == 'DISCARD':

                    discard_tile = action['data']['tile']  # Store discarded tile in history

                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': step + step_offset,
                        'player': action['data']['player'],
                        'action': DISCARD if not next_discard_is_reach else REACH,  # Merged Discard with Reach
                        'tile': discard_tile  # Previous discard pile
                    })

                    next_discard_is_reach = False

                elif tag == 'CALL':

                    meld = action['data']['mentsu']
                    call_type = action['data']['call_type']

                    if call_type in [CHI, PON, MINKAN]:  # The 3 calls that removes from callees pool
                        rows.append({
                            'log_id': log_json.stem,
                            'round': round_number,
                            'step': step + step_offset,
                            'player': action['data']['callee'],
                            'action': REMOVE,
                            'tile': discard_tile  # Previous discard pile
                        })

                    # Handles Chankan - Robbing a Kan
                    if call_type == PON:
                        pons[action['data']['caller']].update(meld)
                    elif call_type == KAKAN:
                        added_tile = list(set(meld) - pons[action['data']['caller']])
                        assert len(added_tile) == 1, 'MORE THAN ONE ADDED TILE!'
                        previous_kakan_tile = added_tile[0]

                    for tile in sorted(meld):  # Added sorting here to include more predictability
                        rows.append({
                            'log_id': log_json.stem,
                            'round': round_number,
                            'step': step + step_offset,
                            'player': action['data']['caller'],
                            'action': action['data']['call_type'],
                            'tile': tile
                        })

                elif tag == 'REACH' and action['data']['step'] == 1:  # 'REACH' is two-step, we check only for 1st part
                    next_discard_is_reach = True
                    step_offset -= 1

                elif tag == 'DORA':

                    step_offset -= 1  # Sync with prior Kan

                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': step + step_offset,
                        'player': -1,
                        'action': 'Dora',
                        'tile': action['data']['hai']  # Tile revealed
                    })

                elif tag == 'AGARI':

                    is_ron = 'loser' in action['data']  # Check if RON or TSUMO
                    previous_action = rows[-1]['action']

                    if is_ron:  # if RON -> prepend REMOVE action
                        rows.append({
                            'log_id': log_json.stem,
                            'round': round_number,
                            'step': step + step_offset,
                            'player': action['data']['loser'],
                            'action': REMOVE,
                            'tile': discard_tile if previous_action != KAKAN else previous_kakan_tile
                            # Previous discard pile
                        })

                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': step + step_offset,
                        'player': action['data']['winner'],
                        'action': 'Ron' if is_ron else 'Tsumo',
                        'tile': rows[-1]['tile'] if previous_action != KAKAN else previous_kakan_tile
                        # Previous tile (either from DISCARD or DRAW or KAKAN)
                    })

                elif tag == 'RYUUKYOKU':
                    rows.append({
                        'log_id': log_json.stem,
                        'round': round_number,
                        'step': step + step_offset,
                        'player': -1,
                        'action': 'Ryuukyoku',
                        'tile': 255  # Tile revealed
                    })

        df = pd.DataFrame(rows)

        # Managing dtypes (This includes a noticeable overhead about 10-15% in time)
        df = df.astype({
            'log_id': pd.StringDtype(),
            'round': 'uint8',
            'step': 'uint8',
            'player': 'int8',
            'action': 'category',  # dtype `category` will not be saved, and will later be read as `object`
            'tile': 'uint8'
        })

        df.to_parquet(output_path / f"{log_json.stem}.parquet", engine='fastparquet')
