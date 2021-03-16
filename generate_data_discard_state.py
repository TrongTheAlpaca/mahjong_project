import time
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import gc

import pandas as pd

# from timeit import default_timer as timer

import utilities.utilities as util

from pyarrow import parquet

# import torch
# from torch import nn

from typing import List
from utilities.tiles import TilesConverter as tc

pd.set_option('display.max_columns', 500)


# Possible Torch Commands
# torch.cuda.current_device()
# torch.cuda.device(0)
# torch.cuda.device_count()
# torch.cuda.get_device_name(0)
# torch.cuda.is_available()

# Got from https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
# Setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()
#
# # Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
# print()


def bin_string(number, digits=136):
    return "{n:0>{d}b}".format(n=number, d=digits)


def int_to_136_binary(number: int) -> List[int]:
    return [0 if number & 1 << 135 - i == 0 else 1 for i in range(136)]


def int_to_136_array(number: int) -> List[int]:
    """ Parses an int into an 136 tiles array. """
    return [i for i in range(136) if number & ((1 << 135) >> i) > 0]


def int_to_34_indices(number: int) -> List[int]:
    array = []
    for i in range(34):
        for j in range(4):
            if number >> 135 - j - 4 * i & 1:  # Check if current bit equals 1
                array.append(i)
    return array


def int_to_34_array(number: int) -> List[int]:
    """ Parses an int into an 34 tiles array. """
    array = [0] * 34
    for i in range(34):  # Iterate through bits from left-to-right
        for j in range(4):  # 4 iteration per tile
            if number >> 135 - j - 4 * i & 1:  # Check if current bit equals 1
                array[i] += 1
    return array


def column_to_34_nparray(numbers: np.array) -> np.array:
    """ N x 1 Binary Array is expanded to N x 34 Int matrix """

    # Hot bit counts for 4 bits
    bit_lookup = {
        0b0000: 0,
        0b0001: 1,
        0b0010: 1,
        0b0011: 2,
        0b0100: 1,
        0b0101: 2,
        0b0110: 2,
        0b0111: 3,
        0b1000: 1,
        0b1001: 2,
        0b1010: 2,
        0b1011: 3,
        0b1100: 2,
        0b1101: 3,
        0b1110: 3,
        0b1111: 4
    }

    array = np.zeros((len(numbers), 34), dtype=np.int8)
    for row, v in enumerate(tqdm(numbers, position=0, disable=DISABLE_TQDM)):
        for i in range(34):
            array[row, 33 - i] = bit_lookup[(v >> (4 * i)) & 0b1111]  # Right-to-Left Masking
    return array


def int_to_mahjong_string(number):
    """
    Convert integer to the one line mahjong string (akadora=0)
    Example of output:  1244079m3p57z
    """
    return tc.to_one_line_string(int_to_136_array(number), print_aka_dora=True)


# UNUSED
# def flip_bit(bits, flip_index):
#     return bits ^ (1 << 135 - flip_index)


# def turn_on_bit(bits, on_index):
#     return bits | (1 << 135 - on_index)


# def turn_off_bit(bits, off_index):
#     return bits & ~(1 << (135 - off_index))


def generate_hand_state_column(df, player):
    mask = (df.player.values == player) & (df.action.values.isin(['Ron', 'Draw', 'Discard', 'Riichi']))
    call_mask = (df.player.values == player) & (
        df.action.values.isin(['Chi', 'Pon', 'MinKan', 'AnKan', 'KaKan', 'Nuki']))

    hand = df.tile.to_numpy(dtype='object')
    call = df.tile.to_numpy(dtype='object')

    hand[mask] = 1 << (135 - hand[mask])
    hand[~mask] = 0

    call[call_mask] = 1 << (135 - call[call_mask])
    call[~call_mask] = 0

    init_indices = df.index[df['action'] == 'Init'].tolist()
    hands = np.array_split(hand, init_indices)[1:]
    calls = np.array_split(call, init_indices)[1:]

    rounds = []
    for i in trange(len(hands), desc=f"p{player}_hand", disable=DISABLE_TQDM):
        hands[i] = np.bitwise_xor.accumulate(hands[i])
        calls[i] = np.bitwise_or.accumulate(calls[i])

        rounds.append(hands[i] & ~calls[i])

    return np.concatenate(rounds)


def generate_meld_state_column(df, player):
    call_mask = (df.player.values == player) & (
        df.action.values.isin(['Chi', 'Pon', 'MinKan', 'AnKan', 'KaKan', 'Nuki']))
    remove_mask = (df.player.values == player) & (df.action == 'Remove')

    call = df.tile.to_numpy(dtype='object')
    kill = df.tile.to_numpy(dtype='object')

    call[call_mask] = 1 << (135 - call[call_mask])
    call[~call_mask] = 0

    kill[remove_mask] = 1 << (135 - kill[remove_mask])
    kill[~remove_mask] = 0

    init_indices = df.index[df['action'] == 'Init'].tolist()
    calls = np.array_split(call, init_indices)[1:]
    kills = np.array_split(kill, init_indices)[1:]

    rounds = []
    for i in trange(len(calls), desc=f"p{player}_meld", disable=DISABLE_TQDM):
        kills[i] = np.bitwise_or.accumulate(kills[i])
        calls[i] = np.bitwise_or.accumulate(calls[i])

        rounds.append(calls[i] & ~kills[i])

    return np.concatenate(rounds)


def generate_discard_state_column(df, player):
    discard_mask = (df.player.values == player) & ((df.action == 'Discard') | (df.action == 'Riichi'))

    discard = df.tile.to_numpy(dtype='object')

    discard[discard_mask] = 1 << (135 - discard[discard_mask])
    discard[~discard_mask] = 0

    init_indices = df.index[df['action'] == 'Init'].tolist()
    discards = np.array_split(discard, init_indices)[1:]

    rounds = []
    for i in trange(len(discards), desc=f"p{player}_disc", disable=DISABLE_TQDM):
        discards[i] = np.bitwise_or.accumulate(discards[i])

        rounds.append(discards[i])

    return np.concatenate(rounds)


def generate_pool_state_column(df, player):
    discard_mask = (df.player.values == player) & ((df.action == 'Discard') | (df.action == 'Riichi'))
    steal_mask = (df.player.values == player) & (df.action.values.isin(['Remove', 'Chi', 'Pon', 'MinKan']))

    pool = df.tile.to_numpy(dtype='object')
    steal = df.tile.to_numpy(dtype='object')

    pool[discard_mask] = 1 << (135 - pool[discard_mask])
    pool[~discard_mask] = 0

    steal[steal_mask] = 1 << (135 - steal[steal_mask])
    steal[~steal_mask] = 0

    init_indices = df.index[df['action'] == 'Init'].tolist()
    pools = np.array_split(pool, init_indices)[1:]
    steals = np.array_split(steal, init_indices)[1:]

    rounds = []
    for i in trange(len(pools), desc=f"p{player}_pool", disable=DISABLE_TQDM):
        pools[i] = np.bitwise_or.accumulate(pools[i])
        steals[i] = np.bitwise_or.accumulate(steals[i])

        rounds.append(pools[i] & ~steals[i])

    return np.concatenate(rounds)


def generate_dora_state_column(df):
    mask = (df.action == 'Dora')

    dora = df.tile.to_numpy(dtype='object')

    dora[mask] = 1 << (135 - dora[mask])
    dora[~mask] = 0

    init_indices = df.index[df['action'] == 'Init'].tolist()
    doras = np.array_split(dora, init_indices)[1:]

    return np.concatenate(
        [np.bitwise_or.accumulate(doras[i]) for i in trange(len(doras), desc=f"dora", disable=DISABLE_TQDM)])


def generate_wall_state_column(df):
    wall = np.zeros(len(df), dtype=int)
    wall[df.action == 'Init'] = 122
    wall[df.action == 'Draw'] = 1

    init_indices = df.index[df['action'] == 'Init'].tolist()
    walls = np.array_split(wall, init_indices)[1:]

    return np.concatenate(
        [np.subtract.accumulate(walls[i]) for i in trange(len(walls), desc=f"wall", disable=DISABLE_TQDM)])


def generate_riichi_state_column(df, player):
    result = np.zeros(len(df), dtype=np.bool)

    riichi_indices = df.index[(df.player == player) & (df.action == 'Riichi')].tolist()
    end_indices = df.index[df.action.isin(['Ron', 'Tsumo', 'Ryuukyoku'])].tolist()

    end = -1
    for start in riichi_indices:
        while end < start:
            end = end_indices.pop(0)

        # TODO: `start` instead of +1? If we want to include riichi declarations in discard dataset
        result[start + 1:end + 1] = True

    return result


def generate_phase_column(df: pd.DataFrame) -> np.array:
    # Begin with merging all pools together
    phase = df[f'p0_pool'].to_numpy(dtype='object') | \
            df[f'p1_pool'].to_numpy(dtype='object') | \
            df[f'p2_pool'].to_numpy(dtype='object') | \
            df[f'p3_pool'].to_numpy(dtype='object')

    # Translate merged pool into phases
    for i, x in enumerate(phase):
        ones = bin(int(x)).count("1")
        if ones <= 24:
            phase[i] = 0  # Early Game
        elif 24 < ones <= 48:
            phase[i] = 1  # Mid Game
        else:
            phase[i] = 2  # End Game

    return phase


def undo_bit(df, player_id, source):
    """ Performant bit flipping via XOR. """
    tile_binary = 1 << (135 - df.tile.to_numpy(dtype='object'))  # Transforms tile values to one-hot-encodings
    return np.where(df['player'].to_numpy() == player_id,  # Undo for discarding player
                    df[f'p{player_id}_{source}'].to_numpy(dtype='object') ^ tile_binary,  # Perform XOR if True
                    df[f'p{player_id}_{source}'].to_numpy(dtype='object'))  # Do nothing if False


def roll_columns(arr: np.array, player_column_index: int, target: int):
    """
    Roll columns to emulate relativeness of player seats compared to player POV.
    NB: This is operation modifies the given array inplace!

    player_column_index: the column index for the player column.
    target: the start index of the 4 columns to be rolled. E.g. target = 5, then column 5,6,7,8 will be rolled.
    """

    # We skip rolling player 0's rows as they are already in correct format
    for player in range(1, 4):
        arr[:, target:target + 4][arr[:, player_column_index] == player] = np.roll(
            arr[arr[:, player_column_index] == player][:, target:target + 4], shift=-player, axis=1)


def chunks(l, n):
    """ Divide iterable l into n-sized batches. """
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


# Options
DISABLE_TQDM = True
# CHUNK_SIZE = 1028
OUTPUT_PATH = "E:/mahjong/discard_datasets"

YEARS = [
    2016,
    2017,
    2018
]

CURRENT_FILE_INDEX = 0

round_data = pd.read_parquet(Path('E:') / 'mahjong' / 'pandas' / 'log_round_meta.parquet', engine='fastparquet')

for year in YEARS:

    (Path(OUTPUT_PATH) / str(year)).mkdir(exist_ok=True, parents=True)  # Create current year's folder

    all_logs = [l for l in Path(f'E:/mahjong/state_data/{year}').iterdir()]
    # chunked = chunks(all_logs, CHUNK_SIZE)

    print(year)
    time.sleep(0.2)
    # chunk_bar = tqdm(chunked, total=len(all_logs) // CHUNK_SIZE, unit=f"batch({CHUNK_SIZE}) ", desc=f"{year}", position=0)
    # chunk_bar = tqdm(all_logs, total=len(all_logs), unit=f"batch({CHUNK_SIZE}) ", desc=f"{year}", position=0)
    chunk_bar = tqdm(all_logs, total=len(all_logs), desc=f"{year}", position=0)

    for chunk in chunk_bar:

        chunk_bar.set_description("{:<40}".format("[Loading DataFrames into Memory]"))

        # print("[Loading DataFrames into Memory]")

        # dfs = parquet.ParquetDataset(chunk, use_legacy_dataset=False).read_pandas().to_pandas()
        dfs = pd.read_parquet(chunk)

        dfs.action = dfs.action.astype('category', copy=False)
        # dfs = dfs.astype({
        #     'log_id': pd.StringDtype(),
        #     'round': 'uint8',
        #     'step': 'uint8',
        #     'player': 'int8',
        #     'action': 'category',  # Category must be reassigned due to how parquet works
        #     'tile': 'uint8'
        # }, copy=False)

        # Get Round Data
        # Use `fastparquet` to preserve categorical data
        # temp_round_data = round_data.copy()[round_data.index.isin(dfs['log_id'], level=0)]
        temp_round_data = round_data.loc[chunk.stem]

        # # Generate State DataFrame
        # ```
        # - DRAW    = 'Draw'
        # - DISCARD = 'Discard'
        # - CHI     = 'Chi'
        # - PON     = 'Pon'
        # - MINKAN  = 'MinKan'  # Open Kan
        # - ANKAN   = 'AnKan'    # Closed Kan
        # - KAKAN   = 'KaKan'    # Added Kan, also called 'Shouminkan'
        # - NUKI    = 'Nuki'      # Declare North dora
        # - REMOVE  = 'Remove'
        # - REACH   = 'Riichi'
        # ```

        # This is where God has abandoned us.

        chunk_bar.set_description("{:<40}".format("[Generating States]"))

        dfs['p0_hand'] = generate_hand_state_column(dfs, 0)
        dfs['p1_hand'] = generate_hand_state_column(dfs, 1)
        dfs['p2_hand'] = generate_hand_state_column(dfs, 2)
        dfs['p3_hand'] = generate_hand_state_column(dfs, 3)

        dfs['p0_meld'] = generate_meld_state_column(dfs, 0)
        dfs['p1_meld'] = generate_meld_state_column(dfs, 1)
        dfs['p2_meld'] = generate_meld_state_column(dfs, 2)
        dfs['p3_meld'] = generate_meld_state_column(dfs, 3)

        dfs['p0_discard'] = generate_discard_state_column(dfs, 0)
        dfs['p1_discard'] = generate_discard_state_column(dfs, 1)
        dfs['p2_discard'] = generate_discard_state_column(dfs, 2)
        dfs['p3_discard'] = generate_discard_state_column(dfs, 3)

        # dfs['p0_pool'] = generate_pool_state_column(dfs, 0)
        # dfs['p1_pool'] = generate_pool_state_column(dfs, 1)
        # dfs['p2_pool'] = generate_pool_state_column(dfs, 2)
        # dfs['p3_pool'] = generate_pool_state_column(dfs, 3)
        #
        # dfs['phase'] = generate_phase_column(dfs)

        dfs['p0_riichi'] = generate_riichi_state_column(dfs, 0)
        dfs['p1_riichi'] = generate_riichi_state_column(dfs, 1)
        dfs['p2_riichi'] = generate_riichi_state_column(dfs, 2)
        dfs['p3_riichi'] = generate_riichi_state_column(dfs, 3)

        dfs['dora'] = generate_dora_state_column(dfs)

        dfs['wall'] = generate_wall_state_column(dfs)

        # CREATING THE DISCARD DATASET
        # ----------------------------
        # When the execution of code arrives here, the original DataFrame should be populated with board states.
        # The following steps will alter said states to befit the DISCARD DATASET, which may be wrong in other cases.
        # We will merge the ROUND DATAFRAME together with the DISCARD STATES to construct he final DISCARD DATASET.
        dfs = dfs[(dfs.action == 'Discard') | (dfs.action == 'Riichi')]  # Focus only on the discard/riichi actions

        # Remove cases where the discarding player is in Riichi
        dfs = dfs[((dfs.player == 0) & (~dfs[f'p0_riichi'])) |
                  ((dfs.player == 1) & (~dfs[f'p1_riichi'])) |
                  ((dfs.player == 2) & (~dfs[f'p2_riichi'])) |
                  ((dfs.player == 3) & (~dfs[f'p3_riichi']))]

        dfs = dfs.drop(columns=['step', 'action'])  # We won't be needing these columns anymore
        dfs = dfs.set_index(['log_id', 'round']).join(temp_round_data, how='left')

        indices = list(dfs.columns)  # Get Column Indices before turning DF to np.array, useful for later steps

        # STEP - UNDOING DISCARDING ACTIONS
        # ---------------------------------
        # We filter the logs for states where a player performs a discard action.
        # If we undo the board state to the state before the discard has concluded,
        # we get the state the discarding player had access to before determining which tile to discard.
        #
        # Undoing a discarding tile includes:
        # - Adding the discarded tile back to hand
        # - Remove the discarded tile from discarding player's pool
        # - Remove the discarded tile from discarding player's discarding history

        chunk_bar.set_description("{:<40}".format("[Undoing Discards]"))

        # Undo for all four players
        for row_index in trange(4, disable=DISABLE_TQDM):
            dfs[f'p{row_index}_hand'] = undo_bit(dfs, row_index, 'hand')
            # dfs[f'p{row_index}_pool'] = undo_bit(dfs, row_index, 'pool')
            dfs[f'p{row_index}_discard'] = undo_bit(dfs, row_index, 'discard')

        # STEP - NORMALIZE PLAYER SCORES
        # ------------------------------
        # Effectively making them fit inside np.int8

        chunk_bar.set_description("{:<40}".format("[Normalize Scores]"))

        score_columns = [f'p{i}_start_score' for i in range(4)] + [f'p{i}_end_score' for i in range(4)]
        dfs[score_columns] = np.rint(dfs[score_columns].to_numpy() / 1000)  # Banker's rounding / Round half to even
        dfs[score_columns] = dfs[score_columns].astype('int8')

        # Pandas DataFrame to Torch Tensor
        mega_array = dfs.to_numpy()

        # Emulate Relativeness:
        # ---------------------
        # To make the final data more uniform, we align the data such that it is always relative to current player POV.
        # This entails switching the position of:
        # - players' hands
        # - players' melds
        # - players' pools
        # - players' discards
        # - players' start score
        # - players' end score
        #
        # The rotation is counter-clockwise (as this is more aligned with the actual rules of Riichi Mahjong):
        # - `player 0` = Yourself
        # - `player 1` = *Shimocha* - Opponent on your right
        # - `player 2` = *Toimen* - Opponent across the board
        # - `player 3` = *Kamicha* - Opponent on your left

        chunk_bar.set_description("{:<40}".format("[Roll Columns]"))

        roll_columns(mega_array, indices.index('player'), indices.index('p0_hand'))
        roll_columns(mega_array, indices.index('player'), indices.index('p0_meld'))
        # roll_columns(mega_array, indices.index('player'), indices.index('p0_pool'))
        roll_columns(mega_array, indices.index('player'), indices.index('p0_discard'))
        roll_columns(mega_array, indices.index('player'), indices.index('p0_riichi'))
        roll_columns(mega_array, indices.index('player'), indices.index('p0_start_score'))
        # roll_columns(mega_array, indices.index('player'), indices.index('p0_end_score'))

        # ## Transform Data to Shaped Data
        # This is the step where we pick and select the data we want and organize it into the shape we want.

        # ### Input Data X

        chunk_bar.set_description("{:<40}".format("[Explode State Columns into Matrices]"))

        pov_hand = mega_array[:, indices.index('p0_hand')]  # Due to column roll, p0 is always POV

        array34_pov_hand = column_to_34_nparray(pov_hand)

        array34_melds = np.concatenate(
            [column_to_34_nparray(mega_array[:, indices.index('p0_meld') + i]) for i in range(4)],
            axis=1
        )

        array34_discards = np.concatenate(
            [column_to_34_nparray(mega_array[:, indices.index('p0_discard') + i]) for i in range(4)],
            axis=1
        )

        array34_doras = column_to_34_nparray(mega_array[:, indices.index('dora')])

        metadata = np.column_stack((
            mega_array[:, indices.index('round_wind')],
            mega_array[:, indices.index('dealer')],
            mega_array[:, indices.index('player')],
            mega_array[:, indices.index('honba')],
            mega_array[:, indices.index('riichibo')],
            mega_array[:, indices.index('wall')],

            mega_array[:, indices.index('p0_start_score')],
            mega_array[:, indices.index('p1_start_score')],
            mega_array[:, indices.index('p2_start_score')],
            mega_array[:, indices.index('p3_start_score')],

            mega_array[:, indices.index('p0_riichi')],
            mega_array[:, indices.index('p1_riichi')],
            mega_array[:, indices.index('p2_riichi')],
            mega_array[:, indices.index('p3_riichi')],
        )).astype(np.int8)

        padding = np.full((len(mega_array), 34 - metadata.shape[1]), np.iinfo(np.int8).min, dtype=np.int8)  # Padding

        chunk_bar.set_description("{:<40}".format("[Finalizing Numpy Array Data]"))

        X = np.concatenate(
            [
                metadata,
                padding,
                array34_doras,
                array34_pov_hand,
                array34_melds,
                array34_discards
            ],
            axis=1
        )

        y = mega_array[:, indices.index('tile')].astype(np.uint8) // 4  # Must be uint8 here or it will miscalculate

        A = np.column_stack((X, y))

        chunk_bar.set_description("{:<40}".format("[Saving Array as SciPy Sparse Array to Disk]"))
        # save_npz(Path(OUTPUT_PATH) / f'{CURRENT_FILE_INDEX}.npz', csr_matrix(A).astype(np.int8))
        save_npz(Path(OUTPUT_PATH) / str(year) / f'{chunk.stem}.npz', csr_matrix(A).astype(np.int8))

        # CURRENT_FILE_INDEX += 1

        chunk_bar.set_description("{:<40}".format("Done"))
