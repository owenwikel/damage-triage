"""One colour per damage class, shared by every chart and overlay in the project.

The four real classes were checked with a colour-vision-deficiency validator (every pair,
light background): they stay distinguishable for red/green colourblind viewers.
"un-classified" is deliberately a neutral gray so it reads as "no answer", not a class.
Charts must still carry a legend or labels; never rely on colour alone.
"""

DAMAGE_COLORS = {
    "no-damage": "#2a78d6",      # blue
    "minor-damage": "#eda100",   # yellow
    "major-damage": "#e34948",   # red
    "destroyed": "#4a3aa7",      # dark violet
    "un-classified": "#898781",  # gray
}

# Chart chrome: text, axes and gridlines stay neutral so the data colours stand out.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
