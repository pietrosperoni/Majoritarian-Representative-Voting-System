import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import spatial

numberParties=10
constituenciesFull=(2247780, #Piemonte 1
	2116136, #Piemonte 2
	3878549, #Lombardia 1
	4300066, #Lombardia 2
	1525536, #Lombardia 3
	1029475, #Trentino Alto Adige
	2923457, #Veneto 1
	1933753, #Veneto 2
	1218985, #Friuli Venezia-Giulia
	1570694, #Liguria
	4342135, #Emilia-Romagna
	3672202, #Toscana
	884268,  #Umbria
	1541319, #Marche
	3997465, #Lazio 1
	1505421, #Lazio 2
	1307309, #Abruzzo
	313660,  #Molise
	3054956, #Campania 1
	2711854, #Campania 2
	4052566, #Puglia
	578036,  #Basilicata
	1959050, #Calabria
	2393438, #Sicilia 1
	2609466, #Sicilia 2
	1639362, #Sardegna
	126806)  #Valle d'Aosta
constituencies=[c/5000 for c in constituenciesFull]
numberConsitutencies=len(constituencies)

isAutonomousConstituency=(False, #Piemonte 1
	False, #Piemonte 2
	False, #Lombardia 1
	False, #Lombardia 2
	False, #Lombardia 3
	False, #Trentino Alto Adige
	False, #Veneto 1
	False, #Veneto 2
	False, #Friuli Venezia-Giulia
	False, #Liguria
	False, #Emilia-Romagna
	False, #Toscana
	False, #Umbria
	False, #Marche
	False, #Lazio 1
	False, #Lazio 2
	False, #Abruzzo
	False, #Molise
	False, #Campania 1
	False, #Campania 2
	False, #Puglia
	False, #Basilicata
	False, #Calabria
	False, #Sicilia 1
	False, #Sicilia 2
	False, #Sardegna
	True)  #Valle d'Aosta


numberSeatsByConstituency=(23, #Piemonte 1
	22, #Piemonte 2
	40, #Lombardia 1
	45, #Lombardia 2
	16, #Lombardia 3
	12, #Trentino Alto Adige
	31, #Veneto 1
	20, #Veneto 2
	13, #Friuli Venezia-Giulia
	16, #Liguria
	45, #Emilia-Romagna
	38, #Toscana
	9,  #Umbria
	16, #Marche
	42, #Lazio 1
	16, #Lazio 2
	14, #Abruzzo
	2,  #Molise
	32, #Campania 1
	28, #Campania 2
	42, #Puglia
	6,  #Basilicata
	20, #Calabria
	25, #Sicilia 1
	27, #Sicilia 2
	17, #Sardegna
	1)  #Valle d'Aosta

PartyList={} #once we have calculated the list of parties we store it here

numberSeats=618
#numberSeats=100

#constituencies=(500, 500, 500)

#isAutonomousConstituency=[True]*len(constituencies)
#isAutonomousConstituency=[False]*len(constituencies)

#isAutonomousConstituency=(True, False, True, False, True, False) 

assert(len(isAutonomousConstituency)==len(constituencies))

ResultNationalCountNationalRest=[0,0]
ResultLocalCountNationalRest=[0,0]
ResultLocalCountLocalRest=[0,0]
ResultNationalVotes=[0,0]

numberConstituencies=len(constituencies)
print "numberConstituencies",numberConstituencies
numberVoters=sum(constituencies)

numberElections=1000

voterConstituency=[]
for i in xrange(numberConstituencies):
    voterConstituency+=[i]*constituencies[i]

numberVoterPerSeat=int(numberVoters)/numberSeats
#numberSeatsByConstituency=[c/numberVoterPerSeat for c in constituencies]
#numberUnusedVotes=[c%numberVoterPerSeat for c in constituencies]

#numberUnusedVotes=[c-(numberSeatsByConstituency[i]*numberVoterPerSeat) for i in xrange(numberConstituencies)]
#print "numberVoterPerSeat",numberVoterPerSeat
#print "numberSeatsByConstituency",numberSeatsByConstituency
#totalSeatsTaken=sum(numberSeatsByConstituency)
#print "totalSeatsTaken",totalSeatsTaken
#print "numberUnusedVotes",numberUnusedVotes
#totalUnusedVotes=sum(numberUnusedVotes)
#print "totalUnusedVotes",totalUnusedVotes
#TotalSeatsFromRests=totalUnusedVotes/numberVoterPerSeat
#print "TotalSeatsFromRests",TotalSeatsFromRests
#TotalVotesLost=totalUnusedVotes%numberVoterPerSeat
#print "TotalVotesLost",TotalVotesLost

#k=1.4142  #the probability to vote for a party goes as (1-d)^k.
k=1.424  #the probability to vote for a party goes as (1-d)^k.
#k=1  #the probability to vote for a party goes as (1-d)^k.
maxBias=0
#maxBias=0


def CalculateConstituencyBias(numberConsitutencies, maxBias):
	constituencyBiases=np.random.rand(numberConsitutencies,2)
	for constituency in xrange(numberConsitutencies):
		constituencyBiases[constituency][0]=(constituencyBiases[constituency][0]-0.5)*maxBias
		constituencyBiases[constituency][1]=(constituencyBiases[constituency][1]-0.5)*maxBias
	return constituencyBiases

def ExtractVotesConstituency(indexedConstituencyBallots, Constituency):
	newBallotsConstituency=np.zeros((1,numberParties+2),int)
	newBallotsNonConstituency=np.zeros((1,numberParties+2),int)
	for t in indexedConstituencyBallots:
		if t.item(-2)==Constituency:
			newBallotsConstituency = np.vstack([newBallotsConstituency, t])
		else:
			newBallotsNonConstituency = np.vstack([newBallotsNonConstituency, t])
	newBallotsConstituency=np.delete(newBallotsConstituency, (0), axis=0)    
	newBallotsNonConstituency=np.delete(newBallotsNonConstituency, (0), axis=0)
	return newBallotsNonConstituency, newBallotsConstituency
def CalculatePartyLists(indexedConstituencyBallots):
	PartyLists={}
	FinalResultsDictionary={}
	for Constituency in xrange(numberConstituencies): #first we check the autonomous constituencies
		if isAutonomousConstituency[Constituency]:
			indexedConstituencyBallots,indexedThisConstituencyBallots=ExtractVotesConstituency(indexedConstituencyBallots, Constituency)
			#votesGiven=np.sum(indexedThisConstituencyBallots, axis=1) #number of ballots with at least one party
			#    print "votesGiven=",votesGiven
		 	#    print "average number of votes given=",np.mean(votesGiven)
		 	#    print "median number of votes given=",int(np.median(votesGiven))
		 	#numberVoters=len(votesGiven)
			finalVotes={} #this doesn't get used at the moment
			finalResults=[0]*numberParties #a dictionary that assigns to each party its number of votes
			winners=[]
			while indexedThisConstituencyBallots.any():
				indexedThisConstituencyBallots, finalVotes, winners, finalResults=ExtractNextWinningPartyIndexed(indexedThisConstituencyBallots,finalVotes, winners, finalResults,2)
			PartyLists[Constituency]=winners
			FinalResultsDictionary[Constituency]=finalResults
	finalVotes={} #this doesn't get used at the moment
	finalResults=[0]*numberParties #a dictionary that assigns to each party its number of votes
	winners=[]
	indexedConstituencyBallotsList=indexedConstituencyBallots[:]
	while indexedConstituencyBallotsList.any():
		indexedConstituencyBallotsList, finalVotes, winners, finalResults=ExtractNextWinningPartyIndexed(indexedConstituencyBallotsList,finalVotes, winners, finalResults,2)
	for Constituency in xrange(numberConstituencies):
		if isAutonomousConstituency[Constituency]==False:
			PartyLists[Constituency]=winners
			indexedConstituencyBallots,indexedThisConstituencyBallots=ExtractVotesConstituency(indexedConstituencyBallots, Constituency)
			FinalResultsDictionary[Constituency]=ExtractKnowingPartyList(indexedThisConstituencyBallots, winners)
	return PartyLists, FinalResultsDictionary, winners
def ExtractKnowingPartyList(indexedBallots,winners):
	"""given a series of ballots, and a list of parties, it assigns the ballots following the party list"""
	finalResults=[0]*numberParties
	for party in winners:
		votesReceived=np.sum(indexedBallots, axis=0).tolist()[0]
		finalResults[party]=votesReceived[party]
		newBallots=np.zeros((1,len(votesReceived)),int)
		for i, ballot in enumerate(indexedBallots):
			if ballot.tolist()[0][party]==0:
				newBallots = np.vstack([newBallots, ballot.tolist()])
		newBallots=np.delete(newBallots, (0), axis=0)    
		indexedBallots=np.matrix(newBallots)
	return finalResults
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
def ExtractEmptyBallotsWithConstituencies(indexedConstituencyBallots):
    numberEmptyBallots=0
    newBallots=np.zeros((1,numberParties+2),int)
    for i, ballot in enumerate(indexedConstituencyBallots):
        if ballot[0,:-2].any():
            newBallots = np.vstack([newBallots, ballot.tolist()])
        else:
            numberEmptyBallots+=1
    newBallots=np.delete(newBallots, (0), axis=0)    
    indexedConstituencyBallots=np.matrix(newBallots)
    return indexedConstituencyBallots, numberEmptyBallots
def CalculateNearestParty(matrixDistances):
    nearestParty=[]
    for i in range(numberVoters):
        m=min(matrixDistances[i])
        p=matrixDistances[i].tolist().index(m)
        nearestParty.append(p)
    return nearestParty
def ExtractNextWinningPartyIndexed(indexedBallots,finalVotes, winners,finalResults,numberIndices=1):
    votesReceived=np.sum(indexedBallots, axis=0).tolist()
    for t in xrange(numberIndices):
        votesReceived[0][numberParties+t]=-1
    ballotsToAssign=max(votesReceived[0])
    party=votesReceived[0].index(ballotsToAssign) #the value of party changes from 1 to n for n parties.
    winners.append(party)
    dictionarybase={}
    if type(finalResults)==type(dictionarybase):
        finalResults[party]=ballotsToAssign  
    else:
    	finalResults[party-1]=ballotsToAssign  
    newBallots=np.zeros((1,numberParties+numberIndices),int)
    for i, ballot in enumerate(indexedBallots):
        if ballot.tolist()[0][party]==0:
            newBallots = np.vstack([newBallots, ballot.tolist()])
        else:
            finalVotes[ballot.tolist()[0][numberParties-1+numberIndices]]=party
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
def AnalyseElectionConstituencies(ballots, voterConstituency):
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

    ConstituencyBallots=np.hstack([ballots,np.transpose(np.matrix(voterConstituency))])
    indexedConstituencyBallots=IndexBallots(ConstituencyBallots)    
    indexedConstituencyBallots, numberEmptyBallots=ExtractEmptyBallotsWithConstituencies(indexedConstituencyBallots)
    numberValidVotes=numberVoters-numberEmptyBallots
    PartyLists, FinalResultsDictionary, generalwinners=CalculatePartyLists(indexedConstituencyBallots)

    return PartyLists, FinalResultsDictionary, generalwinners, numberValidVotes, numberEmptyBallots,finalVotes, votesGiven
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
def CalculatePositions(numberParties, numberVoters, maxBias,positionParties=[]):
    if (positionParties==[]):
        positionParties=np.random.rand(numberParties,2)


    for party in xrange(numberParties):
		positionParties[party][0]=(positionParties[party][0]-0.5)*(1+maxBias)
		positionParties[party][1]=(positionParties[party][1]-0.5)*(1+maxBias)


    positionVoters=np.random.rand(numberVoters,2)
    for voter in xrange(numberVoters):
		positionVoters[voter][0]=positionVoters[voter][0]-0.5
		positionVoters[voter][1]=positionVoters[voter][1]-0.5


    matrixDistancesVotersParties=CalculateMatrixDistances(positionVoters,positionParties)
    nearestParty=CalculateNearestParty(matrixDistancesVotersParties)
    matrixProbabilityVotingParty=CalculateMatrixProbability(matrixDistancesVotersParties,k)
    return matrixProbabilityVotingParty,positionParties,positionVoters, nearestParty




def AssignFixedSeats(constituency, FinalResultsDictionary, numberSeats):
	SeatsByParty=[0]*numberParties
	RestsByParty=[0]*numberParties
	effectiveVotes=sum(FinalResultsDictionary[constituency])
	if numberSeats==1: 
		MX=max(FinalResultsDictionary[constituency])
		party=FinalResultsDictionary[constituency].index(MX)
		SeatsByParty[party]=1
		RestsByParty=FinalResultsDictionary[constituency]
		RestsByParty[party]=0
		SeatAssigned=1
	else:
		effectiveVotes=sum(FinalResultsDictionary[constituency])
		numberVoterPerSeatLocal=float(effectiveVotes)/numberSeats
		for party in xrange(numberParties):
			SeatsByParty[party]=FinalResultsDictionary[constituency][party]/numberVoterPerSeatLocal
			RestsByParty[party]=FinalResultsDictionary[constituency][party]%numberVoterPerSeatLocal
		while sum(SeatsByParty)<numberSeats:
			#print "RestsByParty",RestsByParty
			maxRest=max(RestsByParty)
			party=RestsByParty.index(maxRest)
			SeatsByParty[party]+=1
			RestsByParty[party]=0
	SeatAssigned=sum(SeatsByParty)
	#print "constituency", constituency, "SeatAssigned", SeatAssigned, "seats expected", numberSeats
	return SeatsByParty, RestsByParty
def AssignFirstApproximationSeats(constituency, FinalResultsDictionary, VoterPerSeat):
	SeatsByParty=[0]*numberParties
	RestsByParty=[0]*numberParties
	effectiveVotes=sum(FinalResultsDictionary[constituency])
	if FinalResultsDictionary[constituency]<VoterPerSeat: 
		MX=max(FinalResultsDictionary[constituency])
		party=FinalResultsDictionary[constituency].index(MX)
		SeatsByParty[party]=1
		RestsByParty=FinalResultsDictionary[constituency]
		RestsByParty[party]=0
		SeatAssigned=1
	else:
		for party in xrange(numberParties):
			SeatsByParty[party]=FinalResultsDictionary[constituency][party]/VoterPerSeat
			RestsByParty[party]=FinalResultsDictionary[constituency][party]%VoterPerSeat
	SeatAssigned=sum(SeatsByParty)
	#print "constituency", constituency, "SeatAssigned", SeatAssigned
	return SeatsByParty, RestsByParty
def AssignRemainingSeats(SeatsByPartyDict,restsByPartyDict,numberSeats):
	nationalSeatsByParty=[0]*numberParties
	localSeatsByParty=[0]*numberParties
	for constituency in xrange(len(constituencies)):
		for party in xrange(numberParties):
			localSeatsByParty[party]+=SeatsByPartyDict[constituency][party]
	while sum(localSeatsByParty)+sum(nationalSeatsByParty)<numberSeats:
		maxRestinConstituency=[0]*numberConstituencies
		for c in xrange(numberConstituencies):
			maxRestinConstituency[c]=max(restsByPartyDict[c])
		maxMaxes=max(maxRestinConstituency)
		winnerConstituency=maxRestinConstituency.index(maxMaxes)
		winnerParty=restsByPartyDict[winnerConstituency].index(maxMaxes)
		nationalSeatsByParty[winnerParty]+=1
		restsByPartyDict[winnerConstituency][winnerParty]=0
	return nationalSeatsByParty, localSeatsByParty
def AssignNationalSeats(FinalResults,numberSeats):
	SeatsByParty=[0]*numberParties
	RestsByParty=[0]*numberParties
	effectiveVotes=sum(FinalResults)
	#	print "FinalResults=",FinalResults
	#	print "effectiveVotes=",effectiveVotes
	#	print "numberSeats=",numberSeats

	seatCost=float(effectiveVotes)/numberSeats   #Nota, usiamo un quoziente frazionario
	#seatCost=effectiveVotes/numberSeats
	#	print "seatCost=",seatCost
	for p in xrange(numberParties):
		SeatsByParty[p]=int(FinalResults[p]/seatCost)
		RestsByParty[p]=FinalResults[p]-(SeatsByParty[p]*seatCost)
	#	print "SeatsByParty=",SeatsByParty,"sum=",sum(SeatsByParty)
	#	print "RestsByParty=",RestsByParty
	while sum(SeatsByParty)<numberSeats:
		p=RestsByParty.index(max(RestsByParty))
		SeatsByParty[p]+=1
		RestsByParty[p] =0
	#	print "SeatsByParty=",SeatsByParty,"sum=",sum(SeatsByParty)
	#	print "RestsByParty=",RestsByParty
	
	return SeatsByParty
def ShowElectionResultsNationalCount(FinalResultsDictionary): 
	FinalResults=[0]*numberParties
	for p in xrange(numberParties):
		FinalResults[p]=sum([FinalResultsDictionary[c][p] for c in xrange(numberConstituencies)])

	SeatAssigned=AssignNationalSeats(FinalResults,numberSeats)
	TotalAssigned=sum(SeatAssigned)
	AssignedPercent=[round((float(f)/TotalAssigned)*100,1)for f in SeatAssigned]
	AssignedPercentSorted=sorted(AssignedPercent,reverse=True)
	if AssignedPercentSorted[0]>50: 
		print "*",
		ResultNationalCountNationalRest[1]+=1
	else:
		ResultNationalCountNationalRest[0]+=1
	print "National Count National Rest:    Final Results Sorted and in Percent", AssignedPercentSorted
def ShowElectionResultsLocalRest(FinalResultsDictionary): 
	SeatsByPartyDict={}
	RestsByPartyDict={}

	for constituency in xrange(len(constituencies)):
		SeatsByPartyDict[constituency], RestsByPartyDict[constituency]=AssignFixedSeats(constituency, FinalResultsDictionary, numberSeatsByConstituency[constituency])

	SeatAssigned=[0]*numberParties
	for constituency in xrange(len(constituencies)):
		for party in xrange(numberParties):
			SeatAssigned[party]+=SeatsByPartyDict[constituency][party]
	TotalAssigned=sum(SeatAssigned)
	AssignedPercent=[round((float(f)/TotalAssigned)*100,1)for f in SeatAssigned]
	AssignedPercentSorted=sorted(AssignedPercent,reverse=True)
	if AssignedPercentSorted[0]>50: 
		print "*",
		ResultLocalCountNationalRest[1]+=1
	else:
		ResultLocalCountNationalRest[0]+=1
	print "Local Count Local Rest:    Final Results Sorted and in Percent", AssignedPercentSorted
def ShowElectionResultsNationalRest(FinalResultsDictionary,validVotes,numberSeats): 

	#numberValidVoterPerSeat=int(validVotes)/numberSeats
	numberValidVoterPerSeat=float(validVotes)/numberSeats

	SeatsByPartyDict={}
	RestsByPartyDict={}
	for constituency in xrange(len(constituencies)):
		SeatsByPartyDict[constituency], RestsByPartyDict[constituency]=AssignFirstApproximationSeats(constituency, FinalResultsDictionary, numberValidVoterPerSeat)

	nationalSeatsByParty, localSeatsByParty= AssignRemainingSeats(SeatsByPartyDict,RestsByPartyDict,numberSeats)

	AssignedFinale=[0]*numberParties
	for party in xrange(numberParties):
		AssignedFinale[party]=nationalSeatsByParty[party]+localSeatsByParty[party]

	TotalAssignedAfterRests=sum(AssignedFinale)
	AssignedAfterRestsPercent=[round((float(f)/TotalAssignedAfterRests)*100,1)for f in AssignedFinale]
	finalResultsSortedPercent=sorted(AssignedAfterRestsPercent,reverse=True)
	if finalResultsSortedPercent[0]>50: 
		print "*",
		ResultLocalCountLocalRest[1]+=1
	else:
		ResultLocalCountLocalRest[0]+=1

	print "Local Count National Rest: Final Results Sorted and in Percent", finalResultsSortedPercent


def MoveVotersDependingOnConstituency(positionVoters,constituencyBiases,voterConstituency):
	assert len(positionVoters)==len(voterConstituency)
	for v in xrange(len(positionVoters)):
		positionVoters[v]+=constituencyBiases[voterConstituency[v]]
	return positionVoters



def RunSingleElection(numberParties, numberVoters, numberSeats, k, voterConstituency):
	#print "NumberParties",numberParties
	#print "NumberVoters",numberVoters
	#print "NumberSeats",numberSeats
	#print "k",k
	FinalVotes={}
	colorParties=ColorParties(numberParties)
	matrixProbabilityVotingParty, positionParties, positionVoters, nearestParty=CalculatePositions(numberParties,numberVoters,maxBias)
	constituencyBiases=CalculateConstituencyBias(numberConsitutencies,maxBias)
	positionVoters=MoveVotersDependingOnConstituency(positionVoters,constituencyBiases,voterConstituency)

	ballots=RunElections(matrixProbabilityVotingParty) 

	PartyLists, FinalResultsDictionary, generalwinners, validVotes, numberEmptyBallots, finalVotes, votesGiven=AnalyseElectionConstituencies(ballots, voterConstituency)

	finalResults=[0]*numberParties #a dictionary that assigns to each party its number of votes
	for constituency in xrange(numberConstituencies):
		finalResults=[finalResults[p]+FinalResultsDictionary[constituency][p] for p in xrange(numberParties)]
	#print "final results",finalResults

	assert validVotes==sum(finalResults)

	finalResultsSorted=sorted(finalResults,reverse=True)
	finalResultsSortedPercent=[round((float(f)/validVotes)*100,1)for f in finalResultsSorted]

	finalResultsPercent=[round((float(f)/validVotes)*100,1)for f in finalResults]
	finalResultsSortedPercent=sorted(finalResultsPercent,reverse=True)

	if finalResultsSortedPercent[0]>50: 
		print "*",
		ResultNationalVotes[1]+=1
	else:
		ResultNationalVotes[0]+=1
	print "Votes Final Results: Final Results Sorted and in Percent", finalResultsSortedPercent


	#ResultsSorted   =   np.matrix(finalResultsSortedPercent)
	#ResultsUnsorted =   np.matrix(finalResultsPercent)

	#print PartyLists
	#for constituency in xrange(len(constituencies)):
	#	print "Party List for Constituency",constituency, PartyLists[constituency],
	#	if isAutonomousConstituency[constituency]:	print "autonomous"
	#	else: 										print "sync"

	#for constituency in xrange(len(constituencies)):
	#	print "Final Result for Constituency",constituency, FinalResultsDictionary[constituency],
	#	if isAutonomousConstituency[constituency]:	print "autonomous"
	#	else: 										print "sync"

	ShowElectionResultsLocalRest(FinalResultsDictionary)
	ShowElectionResultsNationalRest(FinalResultsDictionary,validVotes,numberSeats)
	ShowElectionResultsNationalCount(FinalResultsDictionary)
	print 

for r in xrange(numberElections):
	RunSingleElection(numberParties, numberVoters, numberSeats, k, voterConstituency)
	print "Result of ", r," elections:"
	print "proportion of    elections where the first party got more than 50 percent of votes:", (ResultNationalVotes[1]            *100/(r+1)),", below:",(ResultNationalVotes[0]            *100/(r+1))
	print "proportion of    Local Count    Local Rest               above 50 percent of seats:", (ResultLocalCountLocalRest[1]      *100/(r+1)),", below:",(ResultLocalCountLocalRest[0]      *100/(r+1))
	print "percentage of    Local Count National Rest               above 50 percent of seats:", (ResultLocalCountNationalRest[1]   *100/(r+1)),", below:",(ResultLocalCountNationalRest[0]   *100/(r+1))
	print "percentage of National Count National Rest               above 50 percent of seats:", (ResultNationalCountNationalRest[1]*100/(r+1)),", below:",(ResultNationalCountNationalRest[0]*100/(r+1))


print "Result of ", numberElections," elections:"
print "proportion of    elections where the first party got more than 50 percent of votes:", (ResultNationalVotes[1]            *100/(r+1)),", below:",(ResultNationalVotes[0]            *100/(r+1))
print "proportion of    Local Count    Local Rest               above 50 percent of seats:", (ResultLocalCountLocalRest[1]      *100/(r+1)),", below:",(ResultLocalCountLocalRest[0]      *100/(r+1))
print "percentage of    Local Count National Rest               above 50 percent of seats:", (ResultLocalCountNationalRest[1]   *100/(r+1)),", below:",(ResultLocalCountNationalRest[0]   *100/(r+1))
print "percentage of National Count National Rest               above 50 percent of seats:", (ResultNationalCountNationalRest[1]*100/(r+1)),", below:",(ResultNationalCountNationalRest[0]*100/(r+1))


quit()

ShowElectionResults(winners, finalResults, validVotes, emptyBallots, numberVoters, positionParties, positionVoters,finalVotes, votesGiven, nearestParty)

for i in range(numberElections):
    #positionParties=[]
    matrixProbabilityVotingParty, positionParties, positionVoters, nearestParty=CalculatePositions(numberParties, numberVoters, positionParties)
    
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

  



