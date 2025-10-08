# Splits
## Overview
These JSON files contain the IDs of the training, validation, and test sets for the design and specificity models.

## Evaluation Valid vs. Valid?
For both the design and specificity models, the evaluation_valid is a subset of the valid set (subsetting criteria discussed in the *Methods* section in the associated paper). The same relationship holds true for the evaluation_test and test set. All model evaluation, aside from the original training curves, is conducted on the "evaluation" subsets.

## Design Set Format
For the design model, we provide a JSON containing the PDB IDs used for training and evaluating the model.

## Specificity Set Format
For the specificity model, we provide a JSON containing a list of (ID, PPM IDs) pairs. The ID is the PDB ID for structures coming from the PDB or a unique identifier for distillation structures. The PPM IDs, of the format [[[PPM source, PPM ID], ...], ...], are a list of lists of PPM IDs. Each sub-list of PPMs is treated as "experimentally equivalent" (as described in the associated paper). Each PPM is represented as a two element list of source and ID. For the test set, we provide the distillation structures in the **cis_bp_test_distillation_structures** folder.

## Disclaimer on TRANSFAC Data
Note, for the specificity model, due to licensing agreements, we are not allowed to distribute the TRANSFAC data used for training or evaluation.