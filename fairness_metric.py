from sklearn import metrics
import numpy as np

def get_cm_values(y_true, y_pred, num_classes):
    MCM = metrics.multilabel_confusion_matrix(y_true, y_pred, labels=range(int(num_classes)))
    tn = MCM[:, 0, 0]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    fp = MCM[:, 0, 1]
    return tn, fn, tp, fp

def get_multi_class_dp(y_pred_list, group_list, num_classes):
    num_classes = int(num_classes)
    dp_group1 = np.zeros(num_classes)
    dp_group2 = np.zeros(num_classes)

    for class_idx in range(int(num_classes)):
        dp_group1[class_idx] = (y_pred_list[group_list==0]==class_idx).sum() / (y_pred_list[group_list==0]).size
        dp_group2[class_idx] = (y_pred_list[group_list==1]==class_idx).sum() / (y_pred_list[group_list==1]).size

    dp = np.asarray([group1 - group2 for (group1, group2) in zip(dp_group1, dp_group2)])
    return dp

def _prf_divide(numerator, denominator, zero_division=0):
    """ Performs division and handles divide-by-zero. (Copyed from sklearn)
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1 # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result
    result[mask] = 0.0 if zero_division in [0] else 1.0
    return result

def compute_fairness_metrics(label_list, y_pred_list, group_list, zero_division=0):
    num_classes = max(label_list.max(), y_pred_list.max())+1
    tn, fn, tp, fp = get_cm_values(label_list, y_pred_list, num_classes)
    # just want to get the weighted sum
    true_sum = tp + fn
    weights = true_sum
    group0_precision = metrics.precision_score(label_list[group_list==0], y_pred_list[group_list==0], average='macro')#, zero_division=0
    group0_recall = metrics.recall_score(label_list[group_list==0], y_pred_list[group_list==0], average='macro')#, zero_division=0

    group1_precision = metrics.precision_score(label_list[group_list==1], y_pred_list[group_list==1], average='macro')#, zero_division=0
    group1_recall = metrics.recall_score(label_list[group_list==1], y_pred_list[group_list==1], average='macro')#, zero_division=0
    group0_accuracy = metrics.accuracy_score(label_list[group_list==0], y_pred_list[group_list==0])
    group1_accuracy = metrics.accuracy_score(label_list[group_list==1], y_pred_list[group_list==1])
    accuracy = metrics.accuracy_score(label_list, y_pred_list)
    F1 = metrics.f1_score(label_list, y_pred_list, average='macro')
    if sum(group_list==0) > 0:
        tn, fn, tp, fp = get_cm_values(label_list[group_list==0], y_pred_list[group_list==0], num_classes)
        sklearn_TPR_group1 = _prf_divide(tp, tp + fn, zero_division)
        sklearn_TNR_group1 = _prf_divide(tn, tn + fp, zero_division)
        sklearn_FPR_group1 = _prf_divide(fp, tn + fp, zero_division) 
    if sum(group_list==1) > 0:
        tn, fn, tp, fp = get_cm_values(label_list[group_list==1], y_pred_list[group_list==1], num_classes)
        precision = np.array(tp)/(np.array(tp)+np.array(fn))
        recall = np.array(tp)/(np.array(tp)+np.array(fp))
        sklearn_TPR_group2 = _prf_divide(tp, tp + fn, zero_division)
        sklearn_TNR_group2 = _prf_divide(tn, tn + fp, zero_division)
        sklearn_FPR_group2 = _prf_divide(fp, tn + fp, zero_division) 
    # equalized opportunity
    equal_opportunity_gap_y0 = np.average(sklearn_TNR_group1-sklearn_TNR_group2)
    equal_opportunity_gap_y1 = np.average(sklearn_TPR_group1-sklearn_TPR_group2)
    equal_opportunity_gap_y0_abs = np.average(np.abs(sklearn_TNR_group1-sklearn_TNR_group2))
    equal_opportunity_gap_y1_abs = np.average(np.abs(sklearn_TPR_group1-sklearn_TPR_group2))

    # demographic_parity
    demographic_parity_distance = np.sum(get_multi_class_dp(y_pred_list, group_list, num_classes))
    demographic_parity_distance_abs = np.sum(np.abs(get_multi_class_dp(y_pred_list, group_list, num_classes)))

    # equalized odds
    equal_odds = ((sklearn_TPR_group1 - sklearn_TPR_group2) + (sklearn_FPR_group1 - sklearn_FPR_group2)).mean() / 2
    equal_odds_abs = np.abs(((sklearn_TPR_group1 - sklearn_TPR_group2) + (sklearn_FPR_group1 - sklearn_FPR_group2))).mean() / 2
    group0_f1_score = metrics.f1_score(label_list[group_list==0], y_pred_list[group_list==0], average='macro')#, zero_division=0
    group1_f1_score = metrics.f1_score(label_list[group_list==1], y_pred_list[group_list==1], average='macro')#, zero_division=0


    fairness_metric = {'fairness/DP': demographic_parity_distance, 'fairness/EOpp0': equal_opportunity_gap_y0, 
                       'fairness/EOpp1': equal_opportunity_gap_y1, 'fairness/EOdds': equal_odds,
                       'fairness/DP_abs': demographic_parity_distance_abs, 'fairness/EOpp0_abs': equal_opportunity_gap_y0_abs, 
                       'fairness/EOpp1_abs': equal_opportunity_gap_y1_abs, 'fairness/EOdds_abs': equal_odds_abs,
                       'accuracy/group0_recall': group0_recall, 'accuracy/group0_precision': group0_precision, 
                       'accuracy/group0_f1-score': group0_f1_score, 
                       'accuracy/group1_recall': group1_recall, 'accuracy/group1_precision': group1_precision, 
                       'accuracy/group1_f1-score': group1_f1_score ,
                       'group0_accuracy':group0_accuracy,
                       'group1_accuracy':group1_accuracy,
                       'avg/accuracy': accuracy,
                       'avg/F1': F1
                       }

    return group0_f1_score, group1_f1_score, fairness_metric

def compute_accuracy(label_list, y_pred_list, zero_division=0):
    num_classes = max(label_list.max(), y_pred_list.max())+1
    tn, fn, tp, fp = get_cm_values(label_list, y_pred_list, num_classes)
    # just want to get the weighted sum

    total_accuracy = metrics.accuracy_score(label_list, y_pred_list)


    fairness_metric = {'Total_Accuracy': total_accuracy }

    return fairness_metric

def accuracy_metrices(label_list, y_pred_list, sensitive_group_list, sensitive_group_name_list):
    # sensitive_group_name_list[0] : sensitive groups name e.g. gender, races etcs.
    results = {
            '{}_acc'.format(sensitive_group_name_list[0]): metrics.accuracy_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0]),
            '{}_acc'.format(sensitive_group_name_list[1]): metrics.accuracy_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1]),
            '{}_precision'.format(sensitive_group_name_list[0]): metrics.precision_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_precision'.format(sensitive_group_name_list[1]): metrics.precision_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            '{}_recall'.format(sensitive_group_name_list[0]): metrics.recall_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_recall'.format(sensitive_group_name_list[1]): metrics.recall_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            '{}_f1_score'.format(sensitive_group_name_list[0]): metrics.f1_score(label_list[sensitive_group_list==0], y_pred_list[sensitive_group_list==0], average='macro', zero_division=0),
            '{}_f1_score'.format(sensitive_group_name_list[1]): metrics.f1_score(label_list[sensitive_group_list==1], y_pred_list[sensitive_group_list==1], average='macro', zero_division=0),
            'accuracy': metrics.accuracy_score(label_list, y_pred_list),
            'F1': metrics.f1_score(label_list, y_pred_list, average='macro'),
    }
    return results