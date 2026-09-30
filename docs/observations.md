# Observations from the most-damaged tiles

Notes from looking at the before/after figures in `outputs/train/viz/` (made by
`scripts/visualize.py`: the train-split tiles with the most destroyed, major-damage and
minor-damage buildings). These are qualitative first impressions from 10 hand-picked
tiles, not measured results.

## Main finding

**Damage is spatially clustered, with sharp boundaries.** In almost every tile, buildings
with the same damage class sit together in contiguous patches, and the change from one
class to the next happens over a short distance rather than gradually. Some of this
clustering may come from how annotators labelled buildings (for example, assigning one
class to a whole block), not only from the disaster itself.

## Tiles referred to

| tile | disaster type | no-damage | minor | major | destroyed | un-classified |
|---|---|---:|---:|---:|---:|---:|
| hurricane-florence_00000485 | flooding | 0 | 0 | 66 | 0 | 0 |
| hurricane-harvey_00000137 | flooding | 0 | 19 | 233 | 0 | 0 |
| hurricane-harvey_00000509 | flooding | 17 | 157 | 0 | 0 | 0 |
| hurricane-matthew_00000156 | wind | 0 | 188 | 285 | 148 | 20 |
| hurricane-matthew_00000158 | wind | 0 | 94 | 441 | 0 | 0 |
| palu-tsunami_00000118 | tsunami | 294 | 0 | 9 | 1,540 | 0 |
| santa-rosa-wildfire_00000084 | fire | 11 | 0 | 0 | 207 | 0 |
| socal-fire_00001075 | fire | 8 | 3 | 0 | 53 | 0 |

Harvey and Florence were hurricanes, but xBD labels them `flooding`, and the flood water
is what these tiles show.

## Flooding

- **hurricane-florence_00000485:** every building is labelled major damage; none are
  minor or destroyed. This is consistent with water rising to a similar level around every
  building while the roofs stay intact. The xBD damage scale counts a building surrounded
  by water as major damage, which may explain why intact roofs still get the major label.
- **hurricane-harvey_00000137:** a similar pattern. Nearly all residential buildings are
  major damage, but some on the left side have less damage (19 are minor). This is
  possibly because the ground rises from right to left, so the left side flooded less.
  Elevation has not been checked against terrain data.
- **hurricane-harvey_00000509:** most buildings are minor damage, and the no-damage
  buildings sit together on the right side. This is possibly higher ground again, with a
  consistent water level everywhere else.

Across the three flood tiles, each tile is dominated by a single class, and where that
class changes it appears to follow the terrain.

## Wind

- **hurricane-matthew_00000156:** more variable than the flood tiles, which is consistent
  with wind causing the damage rather than water. Every classified building is damaged:
  most are major, but many are minor or destroyed. Each class forms large clusters, which
  is possibly consistent with the path of the strongest winds as the hurricane passed.
- **hurricane-matthew_00000158:** similar clustering, but mostly major damage (441 of 535),
  with the rest minor.

## Tsunami

- **palu-tsunami_00000118:** similar to the flood tiles, but more extreme. The area the
  water reached is clearly visible: 1,540 buildings are destroyed, then there is a sharp
  cut-off to 294 no-damage buildings in one corner, which likely marks where the tsunami
  stopped. Only 9 major-damage buildings sit along that boundary, and there is no minor
  damage at all. This is consistent with a very forceful event: buildings were either
  reached and destroyed, or not reached.

## Fire

- **santa-rosa-wildfire_00000084:** the damage is continuous wherever the fire had fuel to
  catch. Most houses are destroyed, and the few no-damage houses appear in clusters,
  possibly where the fire didn't reach. There is nothing in between: no minor or major
  damage.
- **socal-fire_00001075:** with more spread-out houses, the pattern is slightly
  different. Houses packed closely together and in line with trees were mostly destroyed,
  while more isolated houses had no damage or minor damage. Again there is no major damage.

## Caveats

- These tiles were chosen because they contain the *most* damage of one class, so they
  are the extreme cases. Typical tiles are likely to be less clear-cut.
- One or two tiles per disaster type is not enough to generalise from; the hypotheses
  below are meant to be tested on the full dataset.
- The causes above (terrain, wind path, fuel) are interpretations of the images, not
  things the labels record.

## Hypotheses to test

1. **Random tile splits leak neighbourhood information, so a disaster holdout is needed.**
   If damage is clustered, a tile's neighbours share its damage pattern. A model tested on
   random tiles from disasters it trained on could score well by recognising the
   neighbourhood rather than the damage. The draft split in `docs/split_proposal.md`
   holds out whole disasters for this reason.
2. **Model uncertainty will be highest near damage boundaries.** Most buildings sit inside
   a patch of one class, where the prediction should be easy. The buildings along the edge
   of a patch (such as the 9 major-damage buildings on the Palu boundary) are where the
   visual evidence is most ambiguous and the labels are most likely to disagree.
3. **Damage-class distributions differ by disaster type: fire and tsunami are nearly
   binary, while wind is spread across classes.** This matches the counts for all of each
   disaster (train + test pooled, from `docs/split_proposal.md`): Santa Rosa has 4,435
   destroyed but only 97 minor and 72 major; Palu has 6,203 destroyed, 821 major and 1
   minor; Matthew has 8,887 minor, 2,281 major and 2,826 destroyed.
