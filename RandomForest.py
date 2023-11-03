import pandas as pd


class Node:
  diploid = None
  haploid = None
  instances = None

  def __init__(self, instances):
    self.instances = instances


def stopPartition(node):
  instances = node.instances
  instancesCount = len(instances)
  if instancesCount == 0:
    return True

  diploidCount = instances[instances.category == 'diploid'].shape[0]

  if diploidCount == instancesCount:
    return True
  
  haploidCount = instances[instances.category == 'haploid'].shape[0]

  if haploidCount == instancesCount:
    return True
  
  return False

def defineClass(node):
  instances = node.instances
  diploidCount = instances[instances.category == 'diploid'].shape[0]
  haploidCount = instances[instances.category == 'haploid'].shape[0]
  
  if diploidCount > haploidCount:
    node.diploid = instances
    node.haploid = None

  if haploidCount > diploidCount:
    node.haploid = instances
    node.diploid = None

def defineScore(node, instances):
  diploidCount = instances[instances.category == 'diploid'].shape[0]
  haploidCount = instances[instances.category == 'haploid'].shape[0]
  
  if diploidCount > haploidCount:
    node.diploid = instances
    node.haploid = None

  if haploidCount > diploidCount:
    node.haploid = instances
    node.diploid = None

def printNode(node):
  print('diploid:', node.diploid)
  print('haploid:', node.haploid)
  print('instances:', node.instances)

def runRandoForest(node):
  if stopPartition(node):
    defineClass(node)
    return
  
  attribute = defineScore(node, instances)

def main():
  data = {
    "d12": [420, 380, 390],
    "d13": [50, 40, 45],
    "category": ['haploid', 'diploid', 'haploid']
  }
  df = pd.DataFrame(data)
  tree = Node(df)

  runRandoForest(tree)

  

main()