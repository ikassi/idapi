#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    rootStates = list(theData[:,root])
    for state in rootStates: 
        prior[state] += 1
    prior /= len(rootStates)
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    for row in range(theData.shape[0]):
        cPT[theData[row][varC]][theData[row][varP]] += 1;     
    for i in xrange(cPT.shape[1]): 
        cPT[:, i] /= cPT[:, i].sum()
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
    for row in range(theData.shape[0]):
        jPT[theData[row][varRow]][theData[row][varCol]] += 1;
    jPT /= theData.shape[0];
    return jPT
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    for col in aJPT.T:
        col /= col.sum()
    return aJPT
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    # initialising with the priors
    rootPdf = array(naiveBayes[0],float)
    for d in range(naiveBayes[0].shape[0]):
        for s in range(len(theQuery)):
            rootPdf[d] *= naiveBayes[s+1][theQuery[s]][d]
    if rootPdf.sum() > 0:
        rootPdf /= rootPdf.sum()
    return rootPdf
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
    for a in range(jP.shape[0]):
        rowSum = jP[a, :].sum() #sum of each row
        for b in range(jP.shape[1]):
            colSum = jP[:, b].sum() #sum fo each col
            multRowCol = rowSum * colSum
            if multRowCol != 0:
                prob = jP[a][b]/multRowCol
                if prob != 0:
                    mi += jP[a][b] * math.log(prob, 2)
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
    for i in range(noVariables):
        for j in range(i, noVariables):
            MIMatrix[i][j] = MutualInformation(JPT(theData, i, j, noStates))
            if i != j:
                MIMatrix[j][i] = MIMatrix[i][j]
    return MIMatrix
    

# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
    for i in range(depMatrix.shape[0] - 1):
        for j in range(i + 1, depMatrix.shape[0]):
            depList.append((depMatrix[i][j], i, j))
    depList = sorted(depList, key=lambda trip: trip[0], reverse=True)
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4
# checks if a and b are connected in the adjecency matrix
def isConnected(adj,a,b,visited):
    if a in visited:
        return False
    if(a == b):
        return True
    visited.append(a)
    children = adj[a]
    for i in range(len(children)):
        if(children[i] != 0):
            if isConnected(adj,i,b,visited):
                return True
    return False
    
def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    adjmatrix = zeros((noVariables,noVariables))
    for w,a,b in depList:
        if not isConnected(adjmatrix,a,b,[]):
            spanningTree.append((a,b,w))
            spanningTree.append((b,a,w))
            adjmatrix[a][b] = round(w,4)
            adjmatrix[b][a] = round(w,4)
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)


noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
#print 'Vars:', noVariables, '| Roots:', noRoots, '| States:', noStates, '| DataPoints:', noDataPoints
AppendString("results2.txt","Coursework Two Results by cem13 and ik1410")
AppendString("results2.txt","") #blank line
AppendString("results2.txt","Dependency Matrix for HepatitisC")
MIMatrix = DependencyMatrix(theData, noVariables, noStates)
AppendArray("results2.txt", MIMatrix)
AppendString("results2.txt","Dependency List for HepatitisC")
AppendString("results2.txt","(No involutions or self-reflections are included)")
depList = DependencyList(MIMatrix)
AppendArray("results2.txt", depList)
AppendString("results2.txt","Spanning tree for HepatitisC")
AppendString("results2.txt","(Containing (parent, child, weight) triples)")
spanningTree = SpanningTreeAlgorithm(depList, noVariables)
AppendArray("results2.txt", spanningTree)
