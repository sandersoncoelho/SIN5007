from Classifiers import applyHoldout, runNaiveBayes
from LoaderFeatures import FEATURE_NAMES, getFeaturesAsDataFrame
from PcaUtils import runPCA


def main():
  dataFrame = getFeaturesAsDataFrame()
 
  runPCA(dataFrame)

  train, test = applyHoldout(dataFrame)
  runNaiveBayes(train, test, FEATURE_NAMES, plotCM = True)
  
  # runNaiveBayes(dataFrame, ['CS', 'd68', 'd45'])

main()