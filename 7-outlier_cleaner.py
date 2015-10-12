#!/usr/bin/python
import numpy

def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit(ages, net_worths)
    
    
    differences = numpy.subtract(predictions, net_worths)
    errors = numpy.power(differences, 2)
    rank = errors[errors[:,0].argsort()]
    get90Percent = int(len(errors)*0.9)
    maxErrorAllowed = rank[get90Percent]
    for x in range(0, len(errors)):
        if(errors[x] < maxErrorAllowed):
            cleaned_data.append((ages[x], net_worths[x], errors[x]))

    
    return cleaned_data

