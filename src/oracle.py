import numpy
import scipy.stats as stats

def least_confidence(predictions):
    
    #get the most confident class
    best_class = numpy.amax(predictions, axis=1)

    # and then get the 10 least confident images
    image_indices = best_class.argsort()[:10]
    
    return image_indices

def margin_sampling(predictions):
    
    # first let's grab the two best classes
    sorted_array = numpy.sort(predictions, axis=1)
    best_class = sorted_array[:, -2:]

    # now let's substract them
    # since they are sorted, we are sure that [:, 1] > [:, 0], no negative values should be there
    best_class = numpy.subtract(best_class[:, 1], best_class[:, 0])
    
    # and then the 10 images with the smallest margin
    image_indices = best_class.argsort()[:10]

    return image_indices

def entropy(predictions):

    # first let's calculate entropy for each images
    pred_entropy = []
    for position in range(predictions.shape[0]):
        pred_entropy.append(stats.entropy(predictions[position]))
    
    pred_entropy = numpy.array(pred_entropy)
    
    # and now we take the greatest entropy
    image_indices = pred_entropy.argsort()[-10:]

    return image_indices