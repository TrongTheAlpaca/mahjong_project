# NOTES

## Observations
- A kan is either open or closed
  - if closed: always (B!!B) format
  - if open -> 2 ways:
    1. Pon -> Added (Shouminkan)
       1. Kamicha
       2. Toimen
       3. Shumicha
    2. Direct kan (Daiminkan)
       1. Kamicha
       2. Toimen
       3. Shumicha

## Expected Commandline syntax:
```py
mahjong_draw hand "123p678m999s223h" -o out/poop.svg
mahjong_draw hand "123p 6m 78m" -o out/poop.svg

mahjong_draw add row "123p123h" stack "123p123h" row "123p123h"
mahjong_draw add stack "123p123h"


mahjong_draw discard "1233p678m999s223h" -o out/poop.svg

mahjong_draw pon toimen "1233p678m999s223h" -o out/poop.svg

mahjong_draw kan open "1233p" -o out/poop.svg

mahjong_draw kan closed "1234p" -o out/poop.svg

mahjong_draw kan added kamicha "1234p"

mahjong_draw reset  # Resets current meld

mahjong_draw stack "123p"
mahjong_draw stand "123m"
mahjong_draw -o meld.svg

```

## HAND CREATOR USEFLOW

Each tile function should be in this format:
1. Check if given mahjong string is valid
2. Parse it into intermediate form
   1. Denote aligment (standing/lying/stacked)
3. Placement Logic
4. Figure Creation
5. Output


## Adding a single tile flow (ALTERNATIVE 1):
1. Check if valid string
2. Parse string into intermediate format
3. Add tile to current list
4. Calculate current x
5. Calculate current y
6. Calculate current figure dimensions

## Adding tiles flow (ALTERNATIVE 2):
1. Repeat for each tile:
   1. Check if valid string
   2. Parse string into intermediate format
   3. Add tile to current list
2. From current list:
   1. Calculate x
   2. Calculate y
   3. Calculate figure dimensions

## TILE


`123m`

`456p` 

`123s`

`1234h` = EAST, SOUTH, WEST, NORTH

`567h` = WHITE, GREEN, RED

| 1m                  | 0m    |
| --------------------- | ------- |
| ![](svg/tile_m1!.svg) | ![](svg/tile_m5!r.svg) |

## HOW

1. Download SVG files from: https://commons.wikimedia.org/wiki/Category:SVG_Oblique_illustrations_of_Mahjong_tiles

2. Select only needed tiles and give them better names

3. Use SVGO (`https://github.com/svg/svgo`) to minify SVG files and remove CSS style attributes. Foregoing this step will mess up the color when merging multiple SVG together.

4. USE STYLE_INDEX_DIVERSIFIER

5. Use SVG_Utils (`https://github.com/btel/svg_utils`) to merge together.


## KAN
http://arcturus.su/wiki/Kan

3 apporaches to kan:

    - Open Kan = Daiminkan
      - 3+1 different alignments:
        1. Kamicha, left: (-!!!)
        2. Toimen, across: (!-!!) or (!!-!)
        3. Shimocha, right: (!!!-)
    
    - Added Kan = Shouminkan
      - 3 different alignments:
        1. Kamicha, (=!!)
        2. Toimen, (! = !)
        3. Shimocha, (!!=)

    - Closed Kan = Ankan
      - Simplest, always same layout (0110)

# Known Bugs
- Printing 1-sou after tiles from other suits will make its background white. This can be circumvented by printing 1-sou first.
- Marking the initial tile as riichi tile in discard will not work. This is due to how docopt parses the dash.

# Tile Alignment Existence:
The current svg files included with this project:

```
△: Standing
▷: Lieing
▽: Upside-down
◁: Lieing-Inverted
```

```
h1: △ ▷ ▽ ◁
h2: △ ▷ ▽ ◁
h3: △ ▷ ▽ ◁
h4: △ ▷ ▽ ◁
h5: △ ▷    
h6: △ ▷ ▽ ◁
h7: △ ▷ ▽ ◁
m0: △ ▷ ▽ ◁
m1: △ ▷ ▽ ◁
m2: △ ▷ ▽ ◁
m3: △ ▷ ▽ ◁
m4: △ ▷ ▽ ◁
m5: △ ▷ ▽ ◁
m6: △ ▷ ▽ ◁
m7: △ ▷ ▽ ◁
m8: △ ▷ ▽ ◁
m9: △ ▷ ▽ ◁
p0: △ ▷    
p1: △ ▷    
p2: △ ▷ ▽ ◁
p3: △ ▷ ▽ ◁
p4: △ ▷    
p5: △ ▷    
p6: △ ▷ ▽ ◁
p7: △ ▷ ▽ ◁
p8: △ ▷    
p9: △ ▷ ▽ ◁
s0: △ ▷    
s1: △ ▷ ▽ ◁
s2: △ ▷    
s3: △ ▷ ▽ ◁
s4: △ ▷    
s5: △ ▷    
s6: △ ▷    
s7: △ ▷ ▽ ◁
s8: △ ▷    
s9: △ ▷    
xb: △ ▷    
xu: △    
```