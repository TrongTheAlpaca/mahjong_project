import mahjong_drawer as MD
from mahjong_drawer import Seat

md = MD.MahjongDrawer('mahjong_drawer/svg', 'out')

# STANDING TILES NOTE: WORKS!
# md.add_hand('1234506789m 1234506789p 1234506789s 1234567h', 'tiles_all')
# md.add_hand('1234506789m', 'tiles_man')
# md.add_hand('1234506789p', 'tiles_pin')
# md.add_hand('1234506789s', 'tiles_sou')
# md.add_hand('1234567h', 'tiles_honors')
# md.add_hand('1p1s', 'pair.svg')

# md.add_hand('12h23p',   'test_spaces_0.svg')
# md.add_hand('12h 23p',  'test_spaces_1.svg')
# md.add_hand('12h  23p', 'test_spaces_2.svg')

# md.add_hand('12h bb 23p', 'test_extra_0.svg')
# md.add_hand('12h ?? 23p', 'test_extra_1.svg')
# md.add_hand('12h ?b 23p', 'test_extra_2.svg')


# # CLOSED KAN  # NOTE: WORKS
md.closed_kan('1111p', 'test_kan_closed_1.svg')
md.closed_kan('1234p', 'test_kan_closed_2.svg')
md.closed_kan('5505p', 'test_kan_closed_3.svg')


# # ADDED KAN
# md.add_open_kan('5505p', 'ADDED', Seat.KAMICHA,  "test_kan_open_added_KAMICHA.svg")
# md.add_open_kan('5505p', 'ADDED', Seat.TOIMEN,   "test_kan_open_added_TOIMEN.svg")
# md.add_open_kan('5505p', 'ADDED', Seat.SHIMOCHA, "test_kan_open_added_SHIMOCHA.svg")

# md.add_open_kan('5505p', 'CALLED', Seat.KAMICHA,  "test_kan_open_called_KAMICHA.svg")
# md.add_open_kan('5505p', 'CALLED', Seat.TOIMEN,   "test_kan_open_called_TOIMEN.svg")
# md.add_open_kan('5505p', 'CALLED', Seat.SHIMOCHA, "test_kan_open_called_SHIMOCHA.svg")


#md.create_discard_pile('1111112222223333p', "test_discard_1")
md.create_discard_pile('123p24352m-1h23432h1253s', "test_discard_riichi_1")
md.create_discard_pile('-1h23p24352m23432h1253s', "test_discard_riichi_2")
md.create_discard_pile('1h23p24352m23432h1253s-1h', "test_discard_riichi_3")
md.create_discard_pile('123p24352m-1h-23432h1253s', "test_discard_riichi_4")
