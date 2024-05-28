import numpy as np
import torch
import torch.nn as nn


def calibration_error(predictions, references, n_bins=15):
    predictions, references = np.array(predictions), np.array(references)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(predictions, axis=1)
    predictions = np.argmax(predictions, axis=1)
    if references.ndim > 1:
        # Soft labels
        references = np.argmax(references, axis=1)
    accuracies = predictions == references

    ece = np.zeros(1)
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            mce = np.max([calibration_error, mce])
    return ece[0], mce


def custom_metric(logits, labels, num_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(logits):
        logits_tensor = torch.tensor(logits, dtype=torch.float32).to(device)
    else:
        logits_tensor = logits.to(device)
    norm_1 = torch.norm(logits_tensor, dim=1, p=1).mean(dim=0).item()
    norm_2 = torch.norm(logits_tensor, dim=1, p=2).mean(dim=0).item()
    mean_logits = torch.mean(logits_tensor).item()
    probability = nn.Softmax(dim=1)(logits_tensor).cpu().detach().numpy()
    ece_score, mce_score = calibration_error(probability, labels)
    confidence = np.mean(np.max(probability, axis=1))
    if not torch.is_tensor(labels):
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    else:
        labels_tensor = labels.to(device)
    one_hot_labels = np.eye(num_labels)[labels]
    brier_score = np.mean((probability - one_hot_labels) ** 2)
    nll_func = nn.CrossEntropyLoss()
    nll = nll_func(logits_tensor, labels_tensor).item()
    metrics = {
        "ece": ece_score,
        "mce": mce_score,
        "brier": brier_score,
        "nll": nll,
        "norm_l1": norm_1,
        "norm_l2": norm_2,
        "confidence": confidence,
        "mean_logits": mean_logits,
    }
    return metrics
