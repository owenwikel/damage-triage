# Split decision

## What's the split?
- Train: Guatemala, Harvey, Matthew, Mexico, Palu, SoCal fire
- Val: Florence, Midwest (both floods)
- Test: Michael (wind), Santa Rosa (fire)
- Val-tiles: about 10% of the training disasters' tiles, chosen in spatial blocks

## Why whole disasters?
Damage is clustered: neighbouring buildings often share a damage level. A random tile
split would let the model see the same neighbourhoods in training and testing, so scores
would look better than they really are. The real use is a disaster the model has never
seen, so the test should be one too.

## Why val-tiles?
Val (Florence + Midwest) is floods only, with just 372 minor-damage and 187 destroyed
buildings, too few to tune or calibrate the model on. Val-tiles holds out ~10% of the
training disasters' tiles, so it has every damage level and disaster type. I use it to
decide when to stop training and to calibrate confidence. Tiles are held out in spatial
blocks so neighbours stay together.

## Why Michael stays in test
About 10% of Michael's buildings look possibly mislabelled, the most of any disaster.
Real disaster labels are messy, and swapping Michael out would mean choosing an easier
exam. Results are reported per disaster, so Michael's noise doesn't hide how the model
does on Santa Rosa.

## Known weakness
About half the val-tiles tiles border training tiles, so val-tiles scores will be
optimistic. Comparing them with val (unseen disasters) shows how much the model relies
on familiar surroundings.
