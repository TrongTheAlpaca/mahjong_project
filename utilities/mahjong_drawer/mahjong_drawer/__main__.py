"""
*** Mahjong Draw Documentation ***
Draws Japanese Mahjong tiles into SVG-format.

Usage:
  mdraw    hand <mahjong_string> [-o <path>]
  mdraw discard <mahjong_string> [-o <path>]
  mdraw  closed <mahjong_string> [-o <path>]
  mdraw    chii <mahjong_string> [-o <path>]
  mdraw     pon (shimocha | toimen | kamicha) <mahjong_string> [-o <path>]
  mdraw     kan (shimocha | toimen | kamicha) (called | added) <mahjong_string> [-o <path>]

  mdraw (-h | --help)
  mdraw (-v | --version)

Options:
  -h --help         Show this screen.
  -v --version      Show version.
  <mahjong_string>  Mahjong tiles in string format (e.g. "1234m1234p1234s123h").
  -o <path>         Output to designated filepath.

Structure of <mahjong_string> format:
  Format: ([=][0-9]+[mpsh])+
    =: riichi alignment (only usable when using 'discard')
    0: red five
    m: man
    p: pin
    s: sou
    h: honor

Example Usages:
  mdraw hand 1223m34456p345s22h1p -o example_hand.svg
  mdraw hand "1223m34456p 345s22h 1p" -o example_hand.svg
  mdraw discard 1223m34456p345s-22h -o example_hand.svg
  mdraw closed 1111p -o example_hand.svg
  mdraw chi 456s -o example_hand.svg
  mdraw pon shimocha 5505s
  mdraw kan called toimen 5505s
"""
from .docopt import docopt
from .mahjong_drawer_module import MahjongDrawer
from pathlib import Path


def main():
    args = docopt(__doc__, version="MahjongDraw 0.7")
    string = args["<mahjong_string>"]

    svg_source_path = Path(__file__).parent / "svg"

    md = MahjongDrawer(svg_source_path)

    if args["hand"]:
        md.add_hand(string, args["-o"])

    elif args["chii"]:
        md.add_chii(string, args["-o"])

    elif args["pon"]:
        if args["shimocha"]:
            seat = "shimocha"
        elif args["toimen"]:
            seat = "toimen"
        else:
            seat = "kamicha"

        md.add_pon(string, seat, args["-o"])
    elif args["closed"]:
        md.closed_kan(string, args["-o"])

    elif args["kan"]:
        kan_type = "called" if args["called"] else "added"

        if args["shimocha"]:
            seat = "shimocha"
        elif args["toimen"]:
            seat = "toimen"
        else:
            seat = "kamicha"

        md.add_open_kan(string, kan_type, seat, args["-o"])

    elif args["discard"]:
        md.create_discard_pile(string, args["-o"])

    else:
        print("GIVEN OPTION IS INVALID!")


# https://trstringer.com/easy-and-nice-python-cli/
if __name__ == "__main__":
    main()
