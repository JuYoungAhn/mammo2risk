import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def draw_roc(y_true, y_probas) : 
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_probas)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_probas.ravel())
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
  
def draw_roc_overlay(y_true, y_probas_df) : 
    # Compute ROC curve and ROC area for each class
    color_map = ['black', 'blue', 'red']
    linestyle_map = ['--', ':', '-.']
    label_map = ['Dense', 'Bright', 'Very bright']
    space_map = ['         ','         ',' ']
    plt.figure(figsize=(10,10))
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
        plt.plot(fpr[0], tpr[0], color=color_map[j], linestyle=linestyle_map[j],
                 lw=lw, label=label_map[j]+space_map[j]+'AUC = %0.2f' % roc_auc[0])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-', label='reference')
    plt.show()
    
    return roc_auc[0]