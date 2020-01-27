#Name: Samarth Manjunath
#UTA ID: 1001522809
#Naive Bayes Classifier

#Libraries imported
import numpy as np
import pandas as pd
import scipy.stats
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Values to generate data
mu1 = [1,0]
mu2 = [0,1]
sigma1 = [[1,0.75],[0.75,1]]
sigma2 = [[1,0.75],[0.75,1]]

#training data
train_set0 = np.append(np.random.multivariate_normal(mu1,sigma1,500),np.zeros((500,1)),axis=1)
train_set1 = np.append(np.random.multivariate_normal(mu2,sigma2,500),np.ones((500,1)),axis=1)
X=train_set0+train_set1 #whole training data
Y=["Label0","Label1"]# includes the 2 lables, 0 and 1

#testing data
test_set0 = np.append(np.random.multivariate_normal(mu1,sigma1,500),np.zeros((500,1)),axis=1)
test_set1 = np.append(np.random.multivariate_normal(mu2,sigma2,500),np.ones((500,1)),axis=1)
X_test=test_set0+test_set1 # whole testing data
Y_test=["Label0","Label1"] # includes the 2 labels, 0 and 1

#function to calculate mean
def mean(numbers):
	return sum(numbers)/float(len(numbers))

#function to calculate standard deviation
def stdev(numbers):
	average=mean(numbers)
	variance=sum([pow(x-average,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#summarize dataset
def summarize(dataset):
	summary = [(mean(attribute), stdev(attribute)) for attribute in dataset]
	return summary

#Summary of each attribute
def summarizeByClass(dataset,label):
	separated = dataset
	summaries = {}
	for instances in separated:
		summaries[label] = summarize(dataset)
	return summaries

#summary of train set-0
result_0=summarizeByClass(train_set0,Y[0])

#summary of train set-1
result_1=summarizeByClass(train_set1,Y[1])

merged_result={**result_0,**result_1}


#Gaussian probability density function
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#Calculating the class probability of all attributes
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in merged_result.items():
		probabilities[classValue] = 1
		for i in range(len(inputVector)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

#Make a prediction
def predict(summaries, inputVector):
	
	probabilities = calculateClassProbabilities(merged_result, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel,probabilities

#making predictions for each data instance of dataset
def getPredictions(summaries, testSet):
	predictions = []
	Label0count=0
	Label1count=0
	for i in range(len(test_set0)):
		result = predict(summaries, X_test[i])
		predictions.append(result)
	return predictions

#to calculate accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

#Input for predictions
predictions0=getPredictions(merged_result,test_set0)
predictions1=getPredictions(merged_result,test_set1)
sample0=[]
for i in predictions0:
	sample0.append(i[0])
sample1=[]
for i in predictions1:
	sample1.append(i[0])
count0=sample0.count("Label0")
count1=sample1.count("Label1")
accuracy0=float((count0/len(test_set0)))
accuracy1=float((count1/len(test_set1)))
accuracy=((accuracy0+accuracy1)/2)
predictions=predictions0+predictions1
#output
print("Predictions are",predictions)
print("Accuracy is",float(accuracy*100))
print("Error rate is", (1-accuracy) )
truepositive=(count0+count1)/2
truenegative=((500-count0)+(500-count1))/2
falsepositive=((500-count0)+(500-count1))/2
falsenegative=((500-count0)+(500-count1))/2
recall=truepositive/(truepositive+falsenegative)
print("Recall",recall)
precision=truepositive/(truepositive+falsepositive)
print("Precision",precision)
#confusion matrix
naive0=[]
for i in range(499):
	naive0.append("Label0")
naive1=[]
for i in range(499):
	naive1.append("Label1")
y_actu0 = pd.Series(naive0, name='Actual')
y_pred0 = pd.Series(sample0, name='Predicted')
y_actu1= pd.Series(naive1, name='Actual')
y_pred1=pd.Series(sample1,name='Predicted')
df_confusion0 = pd.crosstab(y_actu0, y_pred0)
df_confusion1 = pd.crosstab(y_actu1, y_pred1)
print("Confusion matrix for 0\n",df_confusion0)
print("Confusion matrix for 1\n",df_confusion1)
#Implementing Scatter plot
xplts=[]
for i in range(500):
	xplts.append(test_set0[i][0])


yplts=[]
for i in range(500):
	yplts.append(test_set0[i][1])

for p,x,y in zip(sample0, xplts, yplts):
        if p == "Label0":
            plt.scatter(x, y, color = 'red')
            continue
        elif p == "Label1":
            plt.scatter(x, y, color = 'blue')
red_patch = mpatches.Patch(color='red', label='Label0')
blue_patch = mpatches.Patch(color='blue', label='Label1')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("xplts")
plt.ylabel("yplts")
plt.show()

#Implementing ROC Curve
naive_bayes_0=[]
naive_bayes_1=[]
for i in range(500):
	if predictions0[i][0]=="Label0":
		naive_bayes_0.append(predictions0[i][1].get("Label0"))
		naive_bayes_1.append(predictions0[i][1].get("Label1"))
	if predictions0[i][0]=="Label1":
		naive_bayes_0.append(predictions0[i][1].get("Label1"))
		naive_bayes_1.append(predictions0[i][1].get("Label0"))


def plot_pdf(good_pdf, bad_pdf, ax):
    ax.fill(good_pdf, "g", alpha=0.5)
    ax.fill(bad_pdf,"r", alpha=0.5)
    ax.set_xlim([0,1])
    ax.set_ylim([0,5])
    ax.set_title("Probability Distribution", fontsize=14)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_xlabel('P(X="bad")', fontsize=12)
    ax.legend(["good","bad"])