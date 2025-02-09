v1.py was used in OLMo experiments as the optimizer for all layers. We receommend starting with this version.


v2.py was used in modded-nanogpt experiments as the optimizer for non first/last layers.

The main differences are 
1. v2.py uses layerwise scaling matching the one used by Muon in modded nanogpt (https://github.com/KellerJordan/modded-nanogpt) while v1.py just scales the update to be sqrt(number of params) norm. 
2. v2.py normalizes the updates for layers on which muon is not being applied to also be qrt(number of params) norm.
