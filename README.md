# EarlyExits

Repo containing the implementation of Early Exit Neural Networks.

The Early Exit Classifiers are designed in two procedures contained in `utils_ee`, `get_intermediate_classifiers_static` and `get_intermediate_classifiers_adaptive`, where the latter automatically designs the classifiers by accounting for computational complexity as described in NACHOS.

The `binary_branch=True` refers to the models designed as shown in [3] where each branch returns two values the logits and the confidence value (the output dimension is #num_classes + 1 for the confidence) while `binary_branch=False` refers to the models used in EDANAS which are standard EECs returning only the logits.

# References

[1] EDANAS: Adaptive Neural Architecture Search for Early Exit Neural Networks, IJCNN 2023 (https://ieeexplore.ieee.org/document/10191876)

[2] NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (https://arxiv.org/abs/2401.13330)

[3] A Probabilistic Re-Intepretation of Confidence Scores in Multi-Exit Models, Entropy 2022 (https://www.mdpi.com/1099-4300/24/1/1)
