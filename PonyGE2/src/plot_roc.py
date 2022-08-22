import pandas as pd
import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erfcinv
#from statsmodels.stats import proportion


def calculate_sensibility_at_level(tpr, fpr, level):
    level_fpr = 1 - level  # fpr is (1-specificity)
    f_sens = interp1d(fpr, tpr)  # interpolate sensibility (tpr = sensibility)
    return (f_sens(level_fpr))


def find_optimal_cutoff(fpr, tpr, thresholds):
    """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        fpr: False positive rate

        tpr : True positive rate

        Returns
        -------
        cutoff value

        """
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))  # Minimum distance to the upper left corner (By Pathagoras' theorem)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    return optimal_sensitivity, optimal_specificity, optimal_threshold


def se_auc(auc, cls):
    """
    Standard error of area
    :param auc: area under the curve
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: standard error
    """
    auc2 = auc ** 2
    q1 = auc / (2 - auc)
    q2 = 2 * auc2 / (1 + auc)
    lu = sum(cls == 1)  # Number of unhealthy subjects (class == 1)
    lh = sum(cls == 0)  # Number of healthy subjects (class == 0)
    V = (auc * (1 - auc) + (lu - 1) * (q1 - auc2) + (lh - 1) * (q2 - auc2)) / (lu * lh)
    se = math.sqrt(V)
    return se


def ci_auc(auc, se, alpha=0.05):
    """
    Confidence interval of AUC
    :param auc: area under the curve
    :param se: standard error
    :param alpha: significance level (default = 0.05)
    :return: confidence interval
    """
    ci_lo = auc + (-1 * math.sqrt(2) * erfcinv(alpha) * se)
    ci_up = auc + (math.sqrt(2) * erfcinv(alpha) * se)
    return ci_lo, ci_up


def ci_sen(optimal_sensitivity, cls):
    """
    Confidence interval of Sensitivity using Simple Asymptotic
    :param optimal_sensitivity: optimal cutoff point
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: confidence interval - array[(low, high)]
    """
    num_u = sum(cls == 1)  # Number of unhealthy subjects (class == 1)
    sa = 1.96 * math.sqrt(optimal_sensitivity * (1 - optimal_sensitivity) / num_u)
    ci_sen = np.zeros(2)
    ci_sen = [optimal_sensitivity - sa, optimal_sensitivity + sa]
    return ci_sen


def ci_spe(optimal_specificity, cls):
    """
        Confidence interval of Specificity using Simple Asymptotic
        :param optimal_specificity: optimal cutoff point
        :param cls: the column of the tag: unhealthy (1) and healthy (0)
        :return: confidence interval - array[(low, high)]
    """
    num_h = sum(cls == 0)  # Number of healthy subjects (class == 0)
    sa = 1.96 * math.sqrt(optimal_specificity * (1 - optimal_specificity) / num_h)
    ci_spe = np.zeros(2)
    ci_spe = [optimal_specificity - sa, optimal_specificity + sa]
    return ci_spe


def plotroc(df, models, levels):

    roc_df = pd.DataFrame(columns=['SensLevel0', 'SensLevel1',
                                   'AUC', 'AucCI_lo', 'AucCI_hi',
                                   'SE',
                                   'OpSen', 'SenCI_lo', 'SenCI_hi',
                                   'OpSpe', 'SpeCI_lo', 'SpeCI_hi'])
    classes = df['class']

    fig =plt.figure(figsize=(8,8))

    for m in models:

        sens = [math.nan, math.nan]  # create a list to hold the sensibility
        model = m['model']  # select the model
        y_proba = df[model]

        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_proba, drop_intermediate=False)
        # Calculate Area under the curve to display on the plot
        # auc = metrics.roc_auc_score(y_test, model.predict(x_test))
        roc_df.loc[model, 'AUC'] = metrics.auc(fpr, tpr)

        # calculate the sensitivity at levels
        roc_df.loc[model, 'SensLevel0'] = calculate_sensibility_at_level(tpr, fpr, levels[0])
        roc_df.loc[model, 'SensLevel1'] = calculate_sensibility_at_level(tpr, fpr, levels[1])

        # Calculate the standard error of AUC
        roc_df.loc[model, 'SE'] = se_auc(roc_df.loc[model, 'AUC'], classes)

        # Calculate the confidence interval of AUC
        roc_df.loc[model, ['AucCI_lo', 'AucCI_hi']] = ci_auc(roc_df.loc[model, 'AUC'], roc_df.loc[model, 'SE'])

        # Calculate the optimal cutoff point, Sensitivity and specificity
        roc_df.loc[model, 'OpSen'], roc_df.loc[model, 'OpSpe'], optimal_threshold = find_optimal_cutoff(fpr, tpr, thresholds)

        # Calculate the confidence interval of Sensitivity
        roc_df.loc[model, ['SenCI_lo', 'SenCI_hi']] = ci_sen(roc_df.loc[model, 'OpSen'], classes)

        # Calculate the confidence interval of Specificity
        roc_df.loc[model, ['SpeCI_lo', 'SpeCI_hi']] = ci_spe(roc_df.loc[model, 'OpSpe'], classes)

        # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        # roc_auc = metrics.auc(fpr, tpr)
        # Now, plot the computed values
        plt.plot(fpr, tpr, label='%s' % (m['label']))
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()  # Display
    #plt.savefig(filename, format='png', dpi=300)
    return roc_df, fig
