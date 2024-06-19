# EarlyExits

Repo containing the implementation of Early Exit Neural Networks.
The Early Exit Classifiers are designed in two procedures contained in utils_ee, get_intermediate_classifiers_static and get_intermediate_classifiers_adaptive, where the latter additionally automatically designs the classifiers by accounting for computational complexity as described in NACHOS.
The binary_branch refers to the models of NACHOS where each branch returns two values the logits and the confidence value (the output dimension is #num_classes + 1 for the confidence) while binary_branch=False refers to the models used in EDANAS which are standard EECs returning only the logits.
