import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import spatial

numberParties=10
numberVoters=5000
numberSeats=100
numberElections=10


#k=1.4142  #the probability to vote for a party goes as (1-d)^k.
k=1.424  #the probability to vote for a party goes as (1-d)^k.



def get_cmap(N,cmap='hsv'):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def ExtractEmptyBallots(indexedBallots,finalVotes):
    numberEmptyBallots=0
    newBallots=np.zeros((1,numberParties+1),int)
    for i, ballot in enumerate(indexedBallots):
        if ballot[0,:-1].any():
            newBallots = np.vstack([newBallots, ballot.tolist()])
        else:
            finalVotes[i]=-1
            numberEmptyBallots+=1
    newBallots=np.delete(newBallots, (0), axis=0)    
    indexedBallots=np.matrix(newBallots)
    return indexedBallots, finalVotes, numberEmptyBallots

def CalculateNearestParty(matrixDistances):
    nearestParty=[]
    for i in range(numberVoters):
        m=min(matrixDistances[i])
        p=matrixDistances[i].tolist().index(m)
        nearestParty.append(p)
    return nearestParty

    
def ExtractNextWinningPartyIndexed(indexedBallots,finalVotes, winners,finalResults):
    votesReceived=np.sum(indexedBallots, axis=0).tolist()
    votesReceived[0][numberParties]=-1
    ballotsToAssign=max(votesReceived[0])
    party=votesReceived[0].index(ballotsToAssign) #the value of party changes from 1 to n for n parties.
    winners.append(party)
    finalResults[party]=ballotsToAssign

    newBallots=np.zeros((1,numberParties+1),int)
    
    for i, ballot in enumerate(indexedBallots):
        if ballot.tolist()[0][party]==0:
            newBallots = np.vstack([newBallots, ballot.tolist()])
        else:
            finalVotes[ballot.tolist()[0][numberParties]]=party
    newBallots=np.delete(newBallots, (0), axis=0)    
    indexedBallots=np.matrix(newBallots)
    return indexedBallots, finalVotes, winners,finalResults


def IndexBallots(ballots):
    indexes=np.transpose(np.matrix(range(numberVoters)))
    indexedBallots=np.hstack([ballots,indexes])
    return indexedBallots


def CalculateMatrixDistances(matRows,matColums):
    matrixDistances=np.zeros((matRows.shape[0],matColums.shape[0]))    
    for r, row in enumerate(matRows):
        for c, column in enumerate(matColums):
            matrixDistances[r,c]=scipy.spatial.distance.euclidean (row,column)
    return matrixDistances

def CalculateMatrixProbability(matrixDistances,k):
    matrixProbability=np.zeros(matrixDistances.shape)
    for (r,c), d in np.ndenumerate(matrixDistances):
        matrixProbability[r,c]=max(1-d,0)**k
    return matrixProbability


def RunElections(matrixProbabilityVotingParty):
    probabilityVotes=np.random.rand(numberVoters,numberParties)
    ballots=np.zeros(probabilityVotes.shape,int)
    for (r,c), p in np.ndenumerate(probabilityVotes):
        ballots[r,c]=1 if p<matrixProbabilityVotingParty[r,c] else 0
    return ballots
    
def AnalyseElection(ballots):
    votesGiven=np.sum(ballots, axis=1) #number of ballots with at least one party
#    print "votesGiven=",votesGiven
#    print "average number of votes given=",np.mean(votesGiven)
#    print "median number of votes given=",int(np.median(votesGiven))

    numberVoters=len(votesGiven)
    finalVotes={}
    finalResults={} #a dictionary that assigns to each party its number of votes
    for t in xrange(numberParties):
        finalResults[t]=0
    winners=[]
    indexedBallots=IndexBallots(ballots)
    
    indexedBallots, finalVotes, numberEmptyBallots=ExtractEmptyBallots(indexedBallots,finalVotes)
    numberValidVotes=numberVoters-numberEmptyBallots
    
    while indexedBallots.any():
        indexedBallots,finalVotes, winners, finalResults=ExtractNextWinningPartyIndexed(indexedBallots,finalVotes, winners, finalResults)
    return winners, finalResults, numberValidVotes, numberEmptyBallots,finalVotes, votesGiven

def ColorParties(numberParties):
    cmap = get_cmap(numberParties+1,'gist_ncar')
    colorParties=[cmap(party) for party in range(numberParties)]
    return colorParties

def ColorVoters(numberParties,finalVotes,colorOthers=(0.0, 0.0, 0.0, 0.0)):
    cmap = get_cmap(numberParties+1,'gist_ncar')
    colorVoters=[]
    for i in range(numberVoters):
        if finalVotes[i]>-1:
            colorVoters.append(cmap(finalVotes[i]))
        else:
            colorVoters.append(colorOthers)
    return colorVoters

def ColorNumberVotes(numberParties,votesGiven):
    cmap = get_cmap(numberParties+1,'gray')
    colorNumberVotes=[cmap(i) for i in votesGiven]
    return colorNumberVotes



def ListSingleVoters(finalVotes,votesGiven):
    singleVoters=[-1]*numberVoters
    for i in xrange(numberVoters):
        if votesGiven[i]==1:
            singleVoters[i]=finalVotes[i]
    return singleVoters

def AreaParty(p): return (p**.5)*500

def ShowElectionResults(winners, finalResults, validVotes, emptyBallots, numberVoters,positionParties,positionVoter,finalVotes,votesGiven, nearestParty):
    print
    print "NEW ELECTION RESULTS"
    print finalResults
    print winners
    print "Empty Ballots=", emptyBallots, round((float(emptyBallots)/numberVoters)*100,1),"%"
    
    singleVoters=ListSingleVoters(finalVotes,votesGiven)
    colorSingleVoters=ColorVoters(numberParties,singleVoters,colorOthers=(1.0,1.0,1.0,1.0))
    
    parliament={}
    percentagesParliament={}
    for p in finalResults.keys():
        parliament[p]=int (round((float(finalResults[p])/validVotes)*numberSeats))
        percentagesParliament[p]=round((float(finalResults[p])/validVotes)*100,1)
    print "\n Parliament=\n",parliament.values(), sum(parliament.values())
    print "\n Parliament in Percentages=\n",percentagesParliament.values()

    for w in winners:
        print "testing w",w
        if parliament[w]:
            print "party %s: %s %s%%"%(w,parliament[w],percentagesParliament[w])

    print "list of Excluded Parties:",
    for w in winners:
        if not parliament[w]:
            print w,
    print

    colorVoters=ColorVoters(numberParties,finalVotes)
    colorNearestVoters=ColorVoters(numberParties,nearestParty)
    
    colorNumberVotes=ColorNumberVotes(numberParties,votesGiven)

    plt.figure(1)

    area=30
    
    plt.scatter(positionVoters[:,0], positionVoters[:,1], s=area, c=colorVoters, alpha=1, linewidths=0)

    area=[AreaParty(p) for p in parliament.values()]
    
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.5)
    
    area=30
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c="black", alpha=1)
#    area=4
#    plt.scatter(positionVoters[:,0], positionVoters[:,1], s=area, c="pink", alpha=0.5)

    plt.figure(2)
    plt.pie(parliament.values(),colors=colorParties)

    plt.figure(3)
    area=30
    plt.scatter(positionVoters[:,0], positionVoters[:,1], s=area, c=colorNumberVotes, alpha=1, linewidths=0)

    area=[AreaParty(p) for p in parliament.values()]
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.3)

    area=100
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.5)

    plt.figure(4)
    area=30
    plt.scatter(positionVoters[:,0], positionVoters[:,1], s=area, c=colorNearestVoters, alpha=1, linewidths=0)
    area=100
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.6)
    
    area=[AreaParty(p) for p in parliament.values()]
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.3)

    plt.figure(5)
    area=100
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.6)
    
        
    area=[AreaParty(p) for p in parliament.values()]
    plt.scatter(positionParties[:,0], positionParties[:,1], s=area, c=colorParties, alpha=0.3)

    area=30
    for i in xrange(numberVoters):
        if singleVoters[i]>-1:
            plt.scatter(positionVoters[i,0], positionVoters[i,1], s=area, c=colorSingleVoters[i], alpha=1)

    plt.show()
    
    
def CalculatePositions(numberParties,numberVoters):
    positionParties=np.random.rand(numberParties,2)
    positionVoters=np.random.rand(numberVoters,2)

    matrixDistancesVotersParties=CalculateMatrixDistances(positionVoters,positionParties)
    nearestParty=CalculateNearestParty(matrixDistancesVotersParties)
    matrixProbabilityVotingParty=CalculateMatrixProbability(matrixDistancesVotersParties,k)
    return matrixProbabilityVotingParty,positionParties,positionVoters, nearestParty
        
print "NumberParties",numberParties
print "NumberVoters",numberVoters
print "NumberSeats",numberSeats
print "k",k

FinalVotes={}

colorParties=ColorParties(numberParties)

matrixProbabilityVotingParty,positionParties,positionVoters, nearestParty=CalculatePositions(numberParties,numberVoters)

ballots=RunElections(matrixProbabilityVotingParty)
winners, finalResults, validVotes, emptyBallots, finalVotes, votesGiven=AnalyseElection(ballots)

finalResultsSorted=sorted(finalResults.values(),reverse=True)
finalResultsSortedPercent=[round((float(f)/validVotes)*100,1)for f in finalResultsSorted]

finalResultsPercent=[round((float(f)/validVotes)*100,1)for f in finalResults.values()]
#finalResultsPercentSorted=sorted(finalResultsPercent,reverse=True)

ResultsSorted   =   np.matrix(finalResultsSortedPercent)
ResultsUnsorted =   np.matrix(finalResultsPercent)

#print
#print ResultsSorted
#print ResultsUnsorted

ShowElectionResults(winners, finalResults, validVotes, emptyBallots, numberVoters, positionParties, positionVoters,finalVotes, votesGiven, nearestParty)

for i in range(numberElections):
    matrixProbabilityVotingParty,positionParties,positionVoters, nearestParty=CalculatePositions(numberParties,numberVoters)
    
    #matrixProbabilityVotingParty=CalculatePositions(numberParties,numberVoters) #comment this line to randomly define the position of the parties and voters each time

    ballots=RunElections(matrixProbabilityVotingParty)
    
    winners, finalResults, validVotes, emptyBallots, finalVotes, votesGiven=AnalyseElection(ballots)
    
    #ShowElectionResults(winners, finalResults, validVotes, emptyBallots,numberVoters,positionParties,positionVoters, finalVotes, votesGiven, nearestParty)
    
    finalResultsSorted=sorted(finalResults.values(),reverse=True)
    finalResultsSortedPercent=[round((float(f)/validVotes)*100,1)for f in finalResultsSorted]

    finalResultsPercent=[round((float(f)/validVotes)*100,1)for f in finalResults.values()]
    #finalResultsPercentSorted=sorted(finalResultsPercent,reverse=True)

    ResultsSorted = np.vstack([ResultsSorted, finalResultsSortedPercent])
    ResultsUnsorted = np.vstack([ResultsUnsorted, finalResultsPercent])
    print i, finalResultsSortedPercent
	

meanResultsSorted   =[round(r,2) for r in   ResultsSorted.mean(axis=0).tolist()[0]]
stdResultsSorted    =[round(r,2) for r in    ResultsSorted.std(axis=0).tolist()[0]]

meanResultsUnsorted =[round(r,2) for r in ResultsUnsorted.mean(axis=0).tolist()[0]]
stdResultsUnsorted  =[round(r,2) for r in  ResultsUnsorted.std(axis=0).tolist()[0]]

for m, s in zip(meanResultsSorted,stdResultsSorted):
    print "%0.1f%%(%0.1f)"%(m,s),
print
for m, s in zip(meanResultsUnsorted,stdResultsUnsorted):
    print "%0.1f%%(%0.1f)"%(m,s),

  








