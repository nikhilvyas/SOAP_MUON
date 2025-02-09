olmo_optimizer.py was used in OLMo experiments as the optimizer for all layers.


nanogpt_optimizer.py was used in modded-nanogpt experiments as the optimizer for non first/last layers. The code for experiments is in the soap-muon-nanogpt folder but it is specific to our cluster. Specifically the configs used in experiments are in soap-muon-nanogpt/configs/exp4_largebatch6_seed_2.yaml and soap-muon-nanogpt/configs/exp4_largebatch6_seed.yaml.

The main differences are 
1. nanogpt_optimizer.py uses layerwise scaling matching the one used by Muon in modded nanogpt (https://github.com/KellerJordan/modded-nanogpt) while olmo_optimizer.py just scales the update to be sqrt(number of params) norm. 
2. olmo_optimizer.py normalizes the updates for layers on which muon is not being applied to also be sqrt(number of params) norm.
