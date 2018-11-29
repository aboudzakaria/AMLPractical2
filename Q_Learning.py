
import random
from random import randint
from time import sleep # used to animate the progress

class Board_game:

    alpha = 0.01 # learning rate for online learning
    T = 10000  # number of episode per game
    # 4 directions
    dx = [-1, +1,  0, 0 ] 
    dy = [ 0,  0, -1, +1]

    def __init__(self,idx = 1,_random_play = False,b=1,e=0.4):
        # open file and read the board
        with open(str(idx+1)+'.data') as f:
            self.board = f.readlines()
        self.board = [row.rstrip('\n') for row in self.board] 

        self.R = len(self.board)    # number of rows of the board 
        self.C = len(self.board[0]) # number of columns
        self.poison_reward = -2*self.C*self.R # poison reward 
        self.treasure_reward = 2*self.C*self.R       # treasure reward
        self.show_progress = False
        # the weights of the Q function that we will learn using online learning
        self.Q_function_w = [[0 for _ in range(3)] for _ in range(4)]
        self.si = 0
        self.sj = 0
        
        self.random_play = _random_play
        self.beta = b    # Q-learning reward parameter
        self.ep = e     # probability for a random move

        self.play_ep = e
        self.learn_ep = e
        self.step_lim = 1000000

    def outside_board(self,i,j):
        return i < 0 or i >= self.R or j < 0 or j >= self.C
    # compute the Q-function at a given state and action
    def Q_function(self,i,j,a):
        W = self.Q_function_w[a]
        return W[0] + W[1]*i + W[2]*j
    # update the weights using online learning
    def update(self,i,j,a,y):
        y_ = self.Q_function(i,j,a)
        delta_y = (y_ - y) 
        #x = [1,i,j]
        self.Q_function_w[a][0] -= self.alpha * delta_y
        self.Q_function_w[a][1] -= self.alpha * delta_y * i
        self.Q_function_w[a][2] -= self.alpha * delta_y * j

    # find the initial position of the agent
    def find_agent(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 'A':
                    return i,j
        return -1,-1

    def board_print(self,_board):
        print('------------------')
        for r in _board:
            for e in r:
                print(e,end='')
            print()
        sleep(0.3) # animation delay

    def learn(self):
        if self.random_play:
            return
        self.si,self.sj = self.find_agent()
        for _ in range(self.T):
            i,j = self.si,self.sj
            gameover = False
            steps_limit = self.step_lim
            while not gameover and steps_limit > 0:
                steps_limit -= 1
                #choose an action 
                if random.uniform(0, 1) <= self.learn_ep:
                    a = randint(0, 3)
                else:
                    temp = [self.Q_function(i,j,q) for q in range(4)]
                    a = temp.index(max(temp)) 
                #move
                #print("a=",a)
                ii,jj = i,j
                i += self.dx[a]
                j += self.dy[a]
                #print("i,j=",i,j)
                #invaild case (outside the board)
                if self.outside_board(i,j):
                    y = 0
                    gameover = True
                #invaild case (on an obstacle)
                elif self.board[i][j] == 'W':
                    y = 0
                    gameover = True
                #terminial state (win)
                elif self.board[i][j] == 'T':
                    y = self.treasure_reward
                    gameover = True
                #terminial state (lose)
                elif self.board[i][j] == 'P':
                    y = self.poison_reward
                    gameover = True
                else:
                    y = -1
                
                if not gameover:
                    temp = [self.Q_function(i,j,q) for q in range(4)]
                    y += self.beta*max(temp)
                if y != 0:
                    self.update(ii,jj,a,y)
    
    def play(self):
        #print('start play')
        gameover = False
        win = False
        i,j = self.si,self.sj
        # make a copy of hte board to use for display
        pb = [list(self.board[q]) for q in range(len(self.board))]
        
        steps_limit = self.step_lim
        while not gameover and steps_limit > 0:
            steps_limit -= 1
            temp = [self.Q_function(i,j,q) for q in range(4)]
            #good = False
            pb[i][j] = 'A'
            if self.show_progress:
                self.board_print(pb)
            
            while True:
                is_random_choice = False
                if self.random_play or random.uniform(0, 1) <= self.play_ep:
                    a = randint(0, 3)
                    is_random_choice = True and (not self.random_play)
                else:
                    a = temp.index(max(temp)) 

                pb[i][j] = '.'
                i += self.dx[a]
                j += self.dy[a]
                
                if self.outside_board(i,j) or self.board[i][j] == 'W' or (self.board[i][j] == 'P' and is_random_choice):
                    i -= self.dx[a]
                    j -= self.dy[a]
                    temp[a] = -1e99
                    continue
                break
                
            
            if self.outside_board(i,j):
                #print('O',end=' ')
                gameover = True
            elif self.board[i][j] == 'W':
                #print('W',end=' ')
                gameover = True
            elif self.board[i][j] == 'T':
                gameover = True
                win = True
            elif self.board[i][j] == 'P':
                #print('P',end=' ')
                gameover = True

        #if win:
            #print("Win!!")
        #else:
        #    #print("GameOver :(")
        
        if self.show_progress:
            sleep(2)
        
        if win:
            return 1
        return 0

if __name__ == '__main__':
    
    number_boards = 13
    repeat_factor = 50

    betarr = [0.1, 0.25, 0.5, 1, 1.5, 2, 5, 10]
    best_accuracy = 0
    best_beta = 0
    
    for b in betarr:
        total_games = 0
        number_wins = 0
        for file_idx in range(number_boards):
            for _ in range(repeat_factor):
                game = Board_game(file_idx,False,b)
                game.learn()
                number_wins += game.play()
                total_games += 1
                #print(total_games)
        
        curr_acc = number_wins/total_games
        print("Beta = ",b,'accuracy = ',end='')
        print(curr_acc)
        if curr_acc > best_accuracy:
            best_accuracy = curr_acc
            best_beta = b
    print('Best accuracy of Best = ',best_accuracy,'Best bata = ',best_beta)

    ep_arr = [.1,.2,.3,.4,.5,.6,.7]
    
    best_accuracy = 0
    best_ep = 0
    for e in ep_arr:
        total_games = 0
        number_wins = 0
        for file_idx in range(number_boards):
            for _ in range(repeat_factor):
                game = Board_game(file_idx,False,b,e)
                game.learn()
                number_wins += game.play()
                total_games += 1
                print(total_games)
        
        curr_acc = number_wins/total_games
        print("Ep = ",e,'accuracy = ',end='')
        print(curr_acc)
        if curr_acc > best_accuracy:
            best_accuracy = curr_acc
            best_ep = b
    print('Best accuracy of Ep = ',best_accuracy,'Best Ep = ',best_beta)

