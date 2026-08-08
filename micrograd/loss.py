
def mse(y_pred, y_gt):
    """ Mean Square Error (MSE) """
    return sum((yi - yi_gt)**2 for yi, yi_gt in zip(y_pred, y_gt))