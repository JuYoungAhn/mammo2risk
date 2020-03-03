import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np

def draw_roc(y_true, y_probas) : 
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_probas)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true, y_probas)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def get_auc(y_true, y_probas) : 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_probas)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc[0]
  
def draw_roc_overlay(y_true, y_probas_df, labels) : 
    
    colors_list = colors = ["#1f77b4","#ff7f0e","#2ca02c","#f61600","#7834a9","#17becf","#684427","#fa5deb","#17becf","#17becf"]
    # linestyle_map = ['--', ':', '-.']
    label_map = labels
    plt.figure(figsize=(10,10))
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    for j in range(0, len(y_probas_df.columns)) : 
        y_probas = y_probas_df.iloc[:,j]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_probas)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probas.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        plt.plot(fpr[0], tpr[0], c=colors_list[j], linestyle='--',
                 lw=lw, label=label_map[j]+'   AUC = %0.2f' % roc_auc[0])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = 20)
        plt.ylabel('True Positive Rate', fontsize = 20)
        plt.legend(loc="lower right", fontsize = 18)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-', label='reference')
    plt.show()
    
    return roc_auc[0]
  
def cm_to_metric(cm, method) : 
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*(PPV*TPR)/(PPV+TPR)
    return pd.DataFrame({'Method':[method], 'Sensitivity':[TPR], 'Specificity':[TNR], 'PPV':[PPV], 'NPV':[NPV],"ACC":[ACC], "F1":[F1]})
  
def add_tertile_columns(df, columns):
    for column in columns:
        new_col = column + "_tertile"
        df[new_col] = pd.qcut(
            df[column], q=[0, 0.333, 0.666, 1], labels=["low", "medium", "high"]
        )
    return df
  
def bland_altman_plot(data1, data2, xlab, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel(xlab)
    plt.ylabel("Discrepancy (%)")
    
def bland_altman_plot2(data1, data2, xlab, *args, **kwargs):
    sns.set_context("paper")
    sns.set(font='serif') 
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(data1, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel(xlab)
    plt.ylabel("Discrepancy (%)")

# https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
def plot_calibration_curve(y, x, name):
    """Plot calibration curve for est w/o and with calibration. """
    fig = plt.figure(figsize=(15, 10))

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x, y, 'o-', color="black")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean Predicted Values")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=16)

    plt.tight_layout()

# https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
def plot_calibration_curve_with_conf(y, x, ub, lb, name):
    """Plot calibration curve for est w/o and with calibration. """
    fig = plt.figure(figsize=(15, 10))

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x, y, 'o-', color="black")
    plt.fill_between(np.arange(0,1,0.1), ub, lb,
                     color='gray', alpha=.5)
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean Predicted Values")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right", fontsize=16)

    plt.tight_layout()