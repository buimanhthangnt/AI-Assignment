# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        positions_of_ghosts = successorGameState.getGhostPositions()

        score = successorGameState.getScore()
        ghostDistance = min(manhattanDistance(newPos, ghost) for ghost in positions_of_ghosts)
        if ghostDistance == 0:
            ghostDistance = 0.0001
        foodDistance = 99999
        if len(newFood.asList()) > 0:
            foodDistance = min(manhattanDistance(newPos, food) for food in newFood.asList())
        return successorGameState.getScore() + 10.0/foodDistance - 10.0/ghostDistance
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax_value(state, depth, agentNum):
            if agentNum == state.getNumAgents():
                depth = depth + 1
                agentNum = 0
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentNum == 0:
                return max_value(state, depth, agentNum)
            else:
                return min_value(state, depth, agentNum)

        def max_value(state, depth, agentNum):
            value = -99999
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value = max(value, minimax_value(successor, depth, agentNum + 1))
            return value

        def min_value(state, depth, agentNum):
            value = 99999
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value = min(value, minimax_value(successor, depth, agentNum + 1))
            return value

        pacmanLegalActions = gameState.getLegalActions(0)
        successors = []
        for action in pacmanLegalActions:
            successors.append(gameState.generateSuccessor(0, action))
        index = 0
        minimax_index = [-99999,0]
        for state in successors:
            temp = minimax_value(state, 0, 1)
            if temp > minimax_index[0]:
                minimax_index[0] = temp
                minimax_index[1] = index
            index = index + 1
        return pacmanLegalActions[minimax_index[1]]
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta_value(state, depth, agentNum, alpha, beta):
            if agentNum == state.getNumAgents():
                depth = depth + 1
                agentNum = 0
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentNum == 0:
                return max_value(state, depth, agentNum, alpha, beta)
            else:
                return min_value(state, depth, agentNum, alpha, beta)

        def max_value(state, depth, agentNum, alpha, beta):
            value = -99999
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value = max(value, alphabeta_value(successor, depth, agentNum + 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state, depth, agentNum, alpha, beta):
            value = 99999
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value = min(value, alphabeta_value(successor, depth, agentNum + 1, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        count = 0
        alpha = -9999
        beta = 9999
        minimax_index = [-99999,0]
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            temp = alphabeta_value(state, 0, 1, alpha, beta)
            if temp > minimax_index[0]:
                minimax_index[0] = temp
                minimax_index[1] = count

            alpha = max(alpha, minimax_index[0])
            count = count + 1
        return gameState.getLegalActions(0)[minimax_index[1]]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax_value(state, depth, agentNum):
            if agentNum == state.getNumAgents():
                depth = depth + 1
                agentNum = 0
            if self.depth == depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentNum == 0:
                return max_value(state, depth, agentNum)
            else:
                return min_value(state, depth, agentNum)

        def max_value(state, depth, agentNum):
            value = -99999
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value = max(value, expectimax_value(successor, depth, agentNum + 1))
            return value

        def min_value(state, depth, agentNum):
            value = 0
            size = len(state.getLegalActions(agentNum))
            for action in state.getLegalActions(agentNum):
                successor = state.generateSuccessor(agentNum, action)
                value += (1.0 / size) * expectimax_value(successor, depth, agentNum + 1)
            return value

        index = 0
        minimax_index = [-99999,0]
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            temp = expectimax_value(state, 0, 1)
            if temp > minimax_index[0]:
                minimax_index[0] = temp
                minimax_index[1] = index
            index = index + 1
        return gameState.getLegalActions(0)[minimax_index[1]]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()

    score = currentGameState.getScore()
    for ghost in currentGameState.getGhostStates():
        distance = manhattanDistance(currentPosition, ghost.getPosition())
        if distance != 0:
            if ghost.scaredTimer > 0:
                score += 100 / distance
            else:
                score -= 10 / distance

    foodDistance = 99999
    if len(currentFood.asList()) > 0:
        foodDistance = min(manhattanDistance(currentPosition, food) for food in currentFood.asList())
    return score + 10.0 / foodDistance
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

