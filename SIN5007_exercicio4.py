from Classifiers import applyHoldout, runNaiveBayes
from DatasetLoader import FEATURE_NAMES, getInstancesAsDataFrame
from PcaUtils import runPCA


def main():
  dataFrame = getInstancesAsDataFrame()
 
  runPCA(dataFrame)

  train, test = applyHoldout(dataFrame)
  runNaiveBayes(train, test, FEATURE_NAMES, plotCM = True)
  
  # runNaiveBayes(dataFrame, ['CS', 'd68', 'd45'])

main()