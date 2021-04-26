from enum import Enum
import re
from pathlib import Path
import svgutils.transform as sg


class MahjongDrawer:
    class Seat(Enum):
        KAMICHA = "kamicha"
        TOIMEN = "toimen"
        SHIMOCHA = "shimocha"

    class MahjongStringError(Exception):
        def __init__(self, expression, message):
            self.expression = expression
            self.message = message

    class _Tiles:
        SLEEPING_Y = -135.624
        STANDING_Y = -146.9503

        TILE_WIDTH = 105
        TILE_HEIGHT = 128

        def __init__(self):
            self.tiles = []

        def add_tile(self, tile_path: str, alignment: str):

            # TILE TUPLE:
            # 0. X
            # 1. Y
            # 2. ALIGNMENT
            # 3. PATH

            if len(self.tiles) == 0:
                if alignment == "STANDING":
                    tile_tuple = (96, self.STANDING_Y, alignment, tile_path)
                    self.tiles.append(tile_tuple)
                elif alignment == "SLEEPING":
                    tile_tuple = (107.202, self.SLEEPING_Y, alignment, tile_path)
                    self.tiles.append(tile_tuple)
                else:
                    raise Exception("ILLEGAL INITIAL ALIGNMENT")
            else:

                prev_x, prev_y, prev_alignment, _ = self.tiles[-1]

                if alignment == "STANDING":
                    x = prev_x + (85 if prev_alignment == "STANDING" else 96)
                    tile_tuple = (x, self.STANDING_Y, alignment, tile_path)
                    self.tiles.append(tile_tuple)
                elif alignment == "SLEEPING":
                    x = prev_x + (100 if prev_alignment == "STANDING" else 105)
                    tile_tuple = (x, self.SLEEPING_Y, alignment, tile_path)
                    self.tiles.append(tile_tuple)
                elif alignment == "STACKING":
                    if prev_alignment == "STANDING":
                        raise Exception(
                            "A STACKING TILE AFTER A STANDING ONE IS NOT POSSIBLE!"
                        )
                    else:
                        tile_tuple = (prev_x, prev_y - 85, alignment, tile_path)
                        self.tiles.append(tile_tuple)

        def add_space(self, alignment="STANDING"):
            if len(self.tiles) == 0:
                if alignment == "STANDING":
                    tile_tuple = (96, self.STANDING_Y, alignment, None)
                    self.tiles.append(tile_tuple)
                elif alignment == "SLEEPING":
                    tile_tuple = (107.202, self.SLEEPING_Y, alignment, None)
                    self.tiles.append(tile_tuple)
                else:
                    raise Exception("ILLEGAL INITIAL ALIGNMENT")

            prev_x, prev_y, prev_alignment, _ = self.tiles[-1]

            if alignment == "STANDING":
                x = prev_x + (85 if prev_alignment == "STANDING" else 96)
                tile_tuple = (x, self.STANDING_Y, alignment, None)
                self.tiles.append(tile_tuple)
            elif alignment == "SLEEPING":
                x = prev_x + (100 if prev_alignment == "STANDING" else 105)
                tile_tuple = (x, self.SLEEPING_Y, alignment, None)
                self.tiles.append(tile_tuple)
            elif alignment == "STACKING":
                if prev_alignment == "STANDING":
                    raise Exception(
                        "A STACKING TILE AFTER A STANDING ONE IS NOT POSSIBLE!"
                    )
                else:
                    tile_tuple = (prev_x, prev_y - 85, alignment, None)
                    self.tiles.append(tile_tuple)

        def calc_highest_stack(self):
            # FIND HIGHEST STACKING
            highest_stack = 0
            current_stack = 0
            for tile in self.tiles:
                _, _, alignment, _ = tile
                if alignment == "STACKING":
                    current_stack += 1
                else:
                    highest_stack = max(current_stack, highest_stack)
                    current_stack = 0

            return max(current_stack, highest_stack)

        def get_figure_tiles(self):

            max_stack = self.calc_highest_stack()
            if max_stack == 0:
                y_offset = 0
            else:
                y_offset = 62.371 + 85 * (max_stack - 1)

            resulting_list = []
            for tile in self.tiles:
                x, y, alignment, path = tile
                if path:  # False if current tile is SPACE
                    tile_figure = sg.fromfile(path).getroot()
                    tile_figure.moveto(x, y + y_offset, scale=0.5)
                    resulting_list.append(tile_figure)

            return resulting_list

        def get_figure_height(self):
            highest_stack = self.calc_highest_stack()
            if 0 < highest_stack:
                return 105 + highest_stack * 85
            else:
                return 127.5

        def get_figure_width(self):

            prev_alignment = self.tiles[0][2]
            width = 106 if prev_alignment == "STANDING" else 128  # Initial Offset

            for tile in self.tiles[1:]:
                _, _, alignment, _ = tile
                if prev_alignment == "STANDING":
                    if alignment == "STANDING":
                        width += 85
                    elif alignment == "SLEEPING":
                        width += 111
                else:
                    if alignment == "STANDING":
                        width += 85
                    elif alignment == "SLEEPING":
                        width += 105

                prev_alignment = alignment

            return width

        def output_to_svg(self, destination_path):
            fig = sg.SVGFigure(
                f"{self.get_figure_width()}px", f"{self.get_figure_height()}px"
            )
            fig.append(self.get_figure_tiles())

            fig.save(destination_path)

    def __init__(self, svg_input_path):

        self.SVG_INPUT = Path(svg_input_path)

        # Standing Tile Measurements
        self.SCALE = 0.5
        self.X_INIT_STAND = 96  # Offset because of svg origin position
        self.X_STAND_WIDTH_FIRST = 105  # svg total width
        self.X_STAND_WIDTH_AFTER = 85  # svg total height

        self.X_LAYING_WIDTH_AFTER_STAND = 100

    @staticmethod
    def _parse_output_name(output_name: str, alternative_name: str):
        if not output_name:
            return alternative_name
        elif not output_name.endswith(".svg"):
            return output_name + ".svg"
        else:
            return output_name

    def create_discard_pile(self, tiles: str, output_name: str):
        # Create so that we can easily create a discard pile svg file.
        # TODO: Adapt Class Tiles here if possible
        # TODO: Make sure that theres no spaces in tiles string
        # TODO: Make sure there's only one riichi tile in pile.

        parsed = self.parse_string_simple(tiles)

        # Find Riichi tile if exists:
        riichi_indices = []
        while "RIICHI" in parsed:
            riichi_ind = parsed.index("RIICHI")
            riichi_indices.append(riichi_ind)
            parsed.remove("RIICHI")

        # Dividing list into n-length sublist, where n = pile_width (SOURCE: https://stackoverflow.com/a/312464)
        pile_width = 6
        rows = [parsed[i : i + pile_width] for i in range(0, len(parsed), pile_width)]

        fig_height = 128 + 105 * (len(rows) - 1)
        fig = sg.SVGFigure("553px", f"{fig_height}px")

        x = self.X_INIT_STAND - (86 if 0 not in riichi_indices else 84)
        y = -146 + 421  # Some magic numbers
        current_index = 0
        for row in rows:
            current_row_svg = []
            for tile in row:
                if current_index in riichi_indices:
                    if Path(self.SVG_INPUT / f"tile_{tile}~.svg").exists():
                        svg = sg.fromfile(
                            self.SVG_INPUT / f"tile_{tile}~.svg"
                        ).getroot()
                    else:
                        # raise "NOT EXISTING SLEEPING INVERSE!"
                        svg = sg.fromfile(
                            self.SVG_INPUT / f"tile_{tile}-.svg"
                        ).getroot()

                    svg.rotate(180)
                    svg.moveto(
                        x + 10, y + 11, scale=self.SCALE
                    )  # Add sleeping tile offset
                    current_row_svg.append(svg)

                    x += 107  # sleeping tile then standing tile difference

                else:
                    if Path(self.SVG_INPUT / f"tile_{tile}i.svg").exists():
                        svg = sg.fromfile(
                            self.SVG_INPUT / f"tile_{tile}i.svg"
                        ).getroot()
                    else:
                        svg = sg.fromfile(
                            self.SVG_INPUT / f"tile_{tile}!.svg"
                        ).getroot()

                    svg.rotate(180)
                    svg.moveto(x, y, scale=self.SCALE)
                    current_row_svg.append(svg)

                    x += self.X_STAND_WIDTH_AFTER

                current_index += 1

            fig.append(list(reversed(current_row_svg)))
            current_row_svg.clear()

            x = self.X_INIT_STAND - 86
            y += 105

        if not output_name:
            output_name = f"discard_{tiles.replace(' ', '_')}.svg"

        if not output_name.endswith(".svg"):
            output_name += ".svg"

        fig.save(f"{output_name}")

    def add_hand(self, tiles: str, output_name: str):
        # Allows: "12h 23m", "12p     "
        # Disallow: "12 23m", "", " ", " 12p", "12p"
        if not re.fullmatch(r"^(?! )((\d+[pmsh]|\?|b) *)+$", tiles):
            raise self.MahjongStringError(
                tiles, "ERROR - Given hand is invalid: " + tiles
            )

        # Parse tiles into proper tile-codes
        parsed = self.parse_string_simple(tiles)

        # Output parsed tile-codes
        _tiles = self._Tiles()
        for tile in parsed:
            if tile == "SPACE":
                _tiles.add_space("STANDING")
            elif tile == "BACKWARD":
                _tiles.add_tile(self.SVG_INPUT / f"tile_xb!.svg", "STANDING")
            elif tile == "UNKNOWN":
                _tiles.add_tile(self.SVG_INPUT / f"tile_xu!.svg", "STANDING")
            else:
                _tiles.add_tile(self.SVG_INPUT / f"tile_{tile}!.svg", "STANDING")

        if not output_name:
            output_name = f"hand_{tiles.replace(' ', '_')}.svg"

        if not output_name.endswith(".svg"):
            output_name += ".svg"

        _tiles.output_to_svg(output_name)

    @staticmethod
    def parse_string_simple(tiles: str) -> list:
        """Parses mahjong string into intermediate format (no alignment).

        Args:
            tiles (str): Mahjong string

        Raises:
            MahjongStringError: If the given mahjong string is invalid.

        Returns:
            list: The intermediate format.
        """
        # Parse tiles into proper tile-codes
        hand = []
        current_codes = []
        for code in tiles:
            if code.isdigit():
                current_codes.append(code)
            elif code == " ":
                hand.append("SPACE")
            elif code == "b":
                hand.append("BACKWARD")
            elif code == "?":
                hand.append("UNKNOWN")
            elif code == "=":
                hand.append("RIICHI")
            elif code in ["p", "m", "s", "h"]:
                # Here: code = suit
                hand.extend([f"{code}{x}" for x in current_codes])
                current_codes.clear()
            else:
                raise Exception(
                    tiles,
                    "ERROR - SHOULD NOT END HERE - Given hand is invalid (?): " + tiles,
                )

        return hand

    def add_pon(self, tiles: str, seat: str, output_name: str):

        parsed = self.parse_string_simple(tiles)
        if len(parsed) != 3:
            raise Exception(f"TOO MANY/FEW TILES!")

        seat = self.Seat(seat)
        if seat not in [self.Seat.KAMICHA, self.Seat.TOIMEN, self.Seat.SHIMOCHA]:
            raise Exception(f"INVALID SEAT CODE: {seat}!")

        _tiles = self._Tiles()
        if seat == self.Seat.KAMICHA:
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}-.svg", "SLEEPING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
        elif seat == self.Seat.TOIMEN:
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}-.svg", "SLEEPING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
        else:  # Seat.SHIMOCHA
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
            _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}-.svg", "SLEEPING")

        if not output_name:
            output_name = f"pon_{tiles.replace(' ', '_')}.svg"

        if not output_name.endswith(".svg"):
            output_name += ".svg"

        _tiles.output_to_svg(output_name)

    def add_chii(self, tiles: str, output_name: str):
        parsed = self.parse_string_simple(tiles)
        if len(parsed) != 3:
            raise Exception(f"TOO MANY/FEW TILES!")

        self.add_pon(tiles, "kamicha", output_name)

    def closed_kan(self, tiles: str, output_name: str):
        """[summary]

        Args:
            tiles (str): [description]
            output_name: str

        Examples:
            tiles = "1111p"

        """

        # TODO: *** Should we restrict it such that the user can only type in one tile?
        # --- Advantage: We can make sure that the user does not input invalid kans.
        # --- Disadvantage: You cannot determine the placement of red 5.

        # TODO: User must not need to worry about alignment, it should be automatic in this function

        # TODO: Compile regex instead?
        if not re.fullmatch(r"^(?! )((\d+[pmsh]|\?|b) *)+(?<! )$", tiles):
            raise self.MahjongStringError(
                tiles, "ERROR - Given hand is invalid: " + tiles
            )

        # Parse tiles into proper tile-codes
        parsed = self.parse_string_simple(tiles)

        if len(parsed) != 4:
            raise Exception(f"TOO MANY/FEW TILES!")

        _tiles = self._Tiles()  # Create tile group to be populated
        _tiles.add_tile(self.SVG_INPUT / f"tile_xb!.svg", "STANDING")
        _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
        _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
        _tiles.add_tile(self.SVG_INPUT / f"tile_xb!.svg", "STANDING")

        if not output_name:
            output_name = f"closed_{tiles.replace(' ', '_')}.svg"

        if not output_name.endswith(".svg"):
            output_name += ".svg"

        _tiles.output_to_svg(output_name)

    def add_open_kan(self, tiles: str, kan_type: str, seat: str, output_name: str):
        """[summary]

        Args:
            kan_type ([type]): [description]
            seat ([type]): [description]
            tiles ([type]): [description]
            output_name: str
        """

        seat = self.Seat(seat)

        if seat not in [self.Seat.KAMICHA, self.Seat.TOIMEN, self.Seat.SHIMOCHA]:
            raise Exception(f"INVALID SEAT CODE: {seat}!")

        if kan_type not in ["added", "called"]:
            raise Exception(f"INVALID KAN TYPE: {kan_type}!")

        parsed = self.parse_string_simple(tiles)

        if len(parsed) != 4:
            raise Exception(f"TOO MANY/FEW TILES!")

        _tiles = self._Tiles()  # Create tile group to be populated

        if kan_type == "added":  # SHOUMINKAN
            if not output_name:
                output_name = f"added_{tiles.replace(' ', '_')}.svg"

            if seat == self.Seat.KAMICHA:
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}-.svg", "SLEEPING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}-.svg", "STACKING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}!.svg", "STANDING")
            elif seat == self.Seat.TOIMEN:
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}-.svg", "SLEEPING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}-.svg", "STACKING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}!.svg", "STANDING")
            else:  # Seat.SHIMOCHA
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}-.svg", "SLEEPING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}-.svg", "STACKING")

        elif kan_type == "called":  # DAIMINKAN
            if not output_name:
                output_name = f"called_{tiles.replace(' ', '_')}.svg"

            if seat == self.Seat.KAMICHA:
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}-.svg", "SLEEPING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}!.svg", "STANDING")
            elif seat == self.Seat.TOIMEN:
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}-.svg", "SLEEPING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}!.svg", "STANDING")
            else:  # Seat.SHIMOCHA
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[0]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[1]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[2]}!.svg", "STANDING")
                _tiles.add_tile(self.SVG_INPUT / f"tile_{parsed[3]}-.svg", "SLEEPING")

        if not output_name.endswith(".svg"):
            output_name += ".svg"

        _tiles.output_to_svg(output_name)
