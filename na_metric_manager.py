import numpy as np
import torch

class MetricManager(object):
    def __init__(self,
                 restype_to_int,
                 weight_metrics,
                 sum_metrics,
                 count_metrics,
                 extra_metrics,
                 dataset_names, 
                 polymer_mask_names, 
                 interface_mask_names):
        self.restype_to_int = restype_to_int
        self.weight_metrics = weight_metrics
        self.sum_metrics = sum_metrics
        self.count_metrics = count_metrics
        self.extra_metrics = extra_metrics
        self.dataset_names = dataset_names
        self.polymer_mask_names = polymer_mask_names
        self.interface_mask_names = interface_mask_names
        
        self.all_mask_names = self.get_all_masks()
        self.mask_to_row = dict(zip(self.all_mask_names, range(len(self.all_mask_names))))
        self.row_to_mask = dict(zip(range(len(self.all_mask_names)), self.all_mask_names))

        self.metric_names = self.weight_metrics + list(self.sum_metrics) + list(map(lambda x: "pred" + x, list(self.count_metrics))) + list(map(lambda x: "true" + x, list(self.count_metrics))) + extra_metrics

        self.metric_to_col = dict(zip(self.metric_names, range(len(self.metric_names))))

        self.metrics = np.zeros((len(self.mask_to_row), 
                                 len(self.metric_to_col)), dtype = np.float64)
    
    def get_all_masks(self):
        all_mask_names = []

        for dataset_name in self.dataset_names:
            for polymer_mask_name in [""] + self.polymer_mask_names:
                for interface_mask_name in [""] + self.interface_mask_names:
                    mask_name = dataset_name
                    if polymer_mask_name != "":
                        mask_name += ("_" + polymer_mask_name)
                    if interface_mask_name != "":
                        mask_name += ("_" + interface_mask_name)
                    all_mask_names.append(mask_name)

        return all_mask_names
    
    def zero_metrics(self):
        self.metrics = np.zeros((len(self.mask_to_row), 
                                 len(self.metric_to_col)), dtype = np.float64)

    def accumulate_metrics_for_mask(self, 
                                    loss, 
                                    accuracy, 
                                    canonical_base_pair_accuracy, 
                                    canonical_base_pair_mask, 
                                    S_true, 
                                    S_pred, 
                                    mask_name, 
                                    mask):
        mask_row = self.mask_to_row[mask_name]
        
        if "weights" in self.weight_metrics:
            weights_col = self.metric_to_col["weights"]
            self.metrics[mask_row, weights_col] += \
                torch.sum(mask).cpu().data.numpy()
        
        if "canonicalBasePairWeights" in self.weight_metrics:
            canonical_base_pair_weights_col = self.metric_to_col["canonicalBasePairWeights"]
            self.metrics[mask_row, canonical_base_pair_weights_col] += \
                torch.sum(mask * canonical_base_pair_mask).cpu().data.numpy()
            
        if "loss" in self.sum_metrics:
            loss_col = self.metric_to_col["loss"]
            self.metrics[mask_row, loss_col] += \
                torch.sum(loss * mask).cpu().data.numpy()
        
        if "accuracy" in self.sum_metrics:
            accuracy_col = self.metric_to_col["accuracy"]
            self.metrics[mask_row, accuracy_col] += \
                torch.sum(accuracy * mask).cpu().data.numpy()
        
        if "canonicalBasePairAccuracy" in self.sum_metrics:
            canonical_base_pair_accuracy_col = self.metric_to_col["canonicalBasePairAccuracy"]
            self.metrics[mask_row, canonical_base_pair_accuracy_col] += \
                torch.sum(canonical_base_pair_accuracy * mask * canonical_base_pair_mask).cpu().data.numpy()
        
        for residue_name in self.count_metrics:
            true_count_col = self.metric_to_col["true" + residue_name]
            self.metrics[mask_row, true_count_col] += \
                torch.sum((S_true == self.restype_to_int[residue_name]).long() * mask)
            
            pred_count_col = self.metric_to_col["pred" + residue_name]
            self.metrics[mask_row, pred_count_col] += \
                torch.sum((S_pred == self.restype_to_int[residue_name]).long() * mask)

    def accumulate(self, 
                   loss, 
                   accuracy, 
                   canonical_base_pair_accuracy, 
                   canonical_base_pair_mask, 
                   S_true, 
                   S_pred, 
                   train_or_valid, 
                   mask_for_loss, 
                   polymer_masks, 
                   interface_masks):
        for polymer_mask_name in [""] + list(polymer_masks.keys()):
            for interface_mask_name in [""] + list(interface_masks.keys()):
                mask_name = train_or_valid
                mask = mask_for_loss

                if polymer_mask_name != "":
                    mask_name += ("_" + polymer_mask_name)
                    mask = mask * polymer_masks[polymer_mask_name]
                if interface_mask_name != "":
                    mask_name += ("_" + interface_mask_name)
                    mask = mask * interface_masks[interface_mask_name]
                
                self.accumulate_metrics_for_mask(loss, 
                                                 accuracy, 
                                                 canonical_base_pair_accuracy, 
                                                 canonical_base_pair_mask, 
                                                 S_true, 
                                                 S_pred, 
                                                 mask_name, 
                                                 mask)
    
    def compute_metrics(self):
        for metric in self.sum_metrics:
            weight_metric = self.sum_metrics[metric]
            weights_col = self.metric_to_col[weight_metric]
            weights = self.metrics[:, weights_col]
            weights_zero_mask = (weights == 0)

            metric_col = self.metric_to_col[metric]

            self.metrics[weights_zero_mask, metric_col] = np.nan
            self.metrics[~weights_zero_mask, metric_col] = self.metrics[~weights_zero_mask, metric_col] / weights[~weights_zero_mask]
        
        for metric in self.count_metrics:
            weight_metric = self.count_metrics[metric]
            weights_col = self.metric_to_col[weight_metric]
            weights = self.metrics[:, weights_col]
            weights_zero_mask = (weights == 0)

            true_metric = "true" + metric
            true_metric_col = self.metric_to_col[true_metric]
            self.metrics[weights_zero_mask, true_metric_col] = np.nan
            self.metrics[~weights_zero_mask, true_metric_col] = self.metrics[~weights_zero_mask, true_metric_col] / weights[~weights_zero_mask]

            pred_metric = "pred" + metric
            pred_metric_col = self.metric_to_col[pred_metric]
            self.metrics[weights_zero_mask, pred_metric_col] = np.nan
            self.metrics[~weights_zero_mask, pred_metric_col] = self.metrics[~weights_zero_mask, pred_metric_col] / weights[~weights_zero_mask]

        # Compute the perplexity. At this point, the loss column of metrics
        # will have already been normalized by the weights.
        if "perplexity" in self.extra_metrics:   
            loss_col = self.metric_to_col["loss"]
            loss = self.metrics[:, loss_col]
            perplexity_col = self.metric_to_col["perplexity"]

            self.metrics[:, perplexity_col] = np.exp(loss)
        
    def create_print_string(self, e, step, train_time, valid_time):
        output_string = f"epoch: {e+1}, step: {step}, train_time: {train_time}, valid_time: {valid_time}"

        for mask_row in range(len(self.row_to_mask)):
            mask_name = self.row_to_mask[mask_row]

            for metric in self.metric_names:
                metric_col = self.metric_to_col[metric]
                data = np.format_float_positional(np.float32(self.metrics[mask_row, metric_col]), unique=False, precision=3)

                output_string += (f", {mask_name}_{metric}: {data}")

        return output_string

def generate_metric_manager(restype_to_int, metrics_to_compute="basic"):
    if metrics_to_compute == "basic":
        dataset_names = ["train", "valid"]
        polymer_mask_names = ["protein", "dna", "rna"]
        weight_metrics = [
            "weights",
            "canonicalBasePairWeights" 
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonicalBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {}
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = []
    elif metrics_to_compute == "all":
        dataset_names = ["train", "valid"]
        polymer_mask_names = ["protein", "dna", "rna"]
        weight_metrics = [
            "weights", 
            "canonicalBasePairWeights"
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonialBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {
            "DA": "weights",
            "DC": "weights",
            "DG": "weights",
            "DT": "weights",
            "A": "weights",
            "C": "weights",
            "G": "weights",
            "U": "weights"
        }
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = ["interface", "nonInterface"]
    elif metrics_to_compute == "na_only_inference":
        dataset_names = ["valid"]
        polymer_mask_names = ["dna", "rna"]
        weight_metrics = [
            "weights", 
            "canonicalBasePairWeights"
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonicalBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {
            "DA": "weights",
            "DC": "weights",
            "DG": "weights",
            "DT": "weights",
            "A": "weights",
            "C": "weights",
            "G": "weights",
            "U": "weights"
        }
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = []

    metric_manager = MetricManager(restype_to_int,
                                   weight_metrics,
                                   sum_metrics,
                                   count_metrics,
                                   extra_metrics,
                                   dataset_names,
                                   polymer_mask_names,
                                   interface_mask_names)
    return metric_manager
