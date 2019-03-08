def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for j in range(prediction.shape[0]):
        if prediction[j] == True and ground_truth[j] == True:
            TP += 1
        elif prediction[j] == True and ground_truth[j] == False:
            FP += 1
        elif prediction[j] == False and ground_truth[j] == True: 
            FN += 1
        elif prediction[j] == False and ground_truth[j] == False:
            TN +=1
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
         
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    acc = 0
    for i in zip(prediction,ground_truth):
        if i[0] == i[1]:
            acc += 1
    acc /= prediction.shape[0]
    return acc
