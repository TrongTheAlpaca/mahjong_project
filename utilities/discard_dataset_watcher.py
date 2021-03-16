# DISCARD DATASET WATCHER
# -----------------------
# This file is used for translating SciPy Sparse Arrays back to human-readable form.


from pathlib import Path
import utilities.tiles as tc

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

DATASET_PATH = "data/2019_discard_dataset_gamewise/"
paths = list(Path(DATASET_PATH).iterdir())

CURRENT_LOG = paths[0]
arr = scipy.sparse.load_npz(CURRENT_LOG).toarray()

print("CURRENT LOG:", CURRENT_LOG.stem)

i_dealer = 1
i_player_pov = 2  # Column Index, not player seat

i_dora = slice(1 * 34, 1 * 34 + 34)

i_pov_hand = slice(2 * 34, 2 * 34 + 34)

i_p0_meld = slice(3 * 34, 3 * 34 + 34)
i_p1_meld = slice(4 * 34, 4 * 34 + 34)
i_p2_meld = slice(5 * 34, 5 * 34 + 34)
i_p3_meld = slice(6 * 34, 6 * 34 + 34)

i_p0_discard = slice(7 * 34, 7 * 34 + 34)
i_p1_discard = slice(8 * 34, 8 * 34 + 34)
i_p2_discard = slice(9 * 34, 9 * 34 + 34)
i_p3_discard = slice(10 * 34, 10 * 34 + 34)

i_p0_score = 6
i_p1_score = 7
i_p2_score = 8
i_p3_score = 9

i_p0_riichi = 10
i_p1_riichi = 11
i_p2_riichi = 12
i_p3_riichi = 13

print("{:<1} - {:<3}- {:>5} - {:>18} | {:>18} - {:>18} - {:>18} - {:>18} | {:>18} - {:>18} - {:>18} - {:>18} | {:>3} - {:>3} - {:>3} - {:>3} | {:>3} - {:>3} - {:>3} - {:>3}"
      .format("D",
              "POV",
              "DORA",
              "POV HAND",
              "P0-MELD",
              "P1-MELD",
              "P2-MELD",
              "P3-MELD",
              "P0-DISCARD",
              "P1-DISCARD",
              "P2-DISCARD",
              "P3-DISCARD",
              "0_$",
              "1_$",
              "2_$",
              "3_$",
              "0_R",
              "1_R",
              "2_R",
              "3_R",
              ))

for row in arr:

    dealer = row[i_dealer]

    player = row[i_player_pov]

    doras = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_dora].tolist()))

    pov_hand = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_pov_hand].tolist()))

    meld_p0 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p0_meld].tolist()))
    meld_p1 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p1_meld].tolist()))
    meld_p2 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p2_meld].tolist()))
    meld_p3 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p3_meld].tolist()))

    discard_p0 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p0_discard].tolist()))
    discard_p1 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p1_discard].tolist()))
    discard_p2 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p2_discard].tolist()))
    discard_p3 = tc.TilesConverter.to_one_line_string(tc.TilesConverter.to_136_array(row[i_p3_discard].tolist()))

    score_p0 = row[i_p0_score]
    score_p1 = row[i_p1_score]
    score_p2 = row[i_p2_score]
    score_p3 = row[i_p3_score]

    riichi_p0 = row[i_p0_riichi]
    riichi_p1 = row[i_p1_riichi]
    riichi_p2 = row[i_p2_riichi]
    riichi_p3 = row[i_p3_riichi]

    print(
        "{:^1} - P{:<1} - {:>5} - {:>18} | {:>18} - {:>18} - {:>18} - {:>18} | {:>18} - {:>18} - {:>18} - {:>18} | {:>3} - {:>3} - {:>3} - {:>3} | {:>3} - {:>3} - {:>3} - {:>3}"
        .format(dealer,
                player,
                doras,
                pov_hand,
                meld_p0,
                meld_p1,
                meld_p2,
                meld_p3,
                discard_p0,
                discard_p1,
                discard_p2,
                discard_p3,
                score_p0,
                score_p1,
                score_p2,
                score_p3,
                riichi_p0,
                riichi_p1,
                riichi_p2,
                riichi_p3,
                ))
