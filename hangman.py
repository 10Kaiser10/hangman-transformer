import numpy as np

class Hangman:
    def __init__(self, dictonary_path, solver, tries=6):
        self.tries = tries
        self.reset_func = solver.reset
        self.guess_func = solver.guess
        with open(dictonary_path) as f:
            self.words = np.array(f.read().splitlines())

    def play(self, n, seed = 42):
        if n > len(self.words):
            print('Max value of n is {0}'.format(len(self.words)))
            return
        
        np.random.seed(seed)
        word_idxs = np.random.permutation(len(self.words))[:n]    
        word_list = list(self.words[word_idxs])

        wins = 0
        game_num = 0
        for word in word_list:
            game_num += 1
            print('Starting Game {0}'.format(game_num))
            self.reset_func()
            outcome = self.play_round(word)
            wins += outcome

        print('Total: {0}, Wins: {1}, Perc: {2}'.format(n, wins, 100*wins/n))

    def play_round(self, word):
        n = len(word)
        curr = "_"*n

        print('You have {0} tries'.format(self.tries))
        tries = self.tries

        while(tries > 0):
            print(curr, ", Tries left {}".format(tries))
            guess = self.guess_func(curr)

            if (word.find(guess) == -1):
                tries -= 1
            elif (word.find(guess) != -1 and curr.find(guess) != -1):
                print('Already attempted')
            else:
                for i in range(n):
                    if word[i] == guess:
                        curr = curr[:i] + guess + curr[i+1:]

            if (curr == word):
                print("Won! Answer: {0}".format(word))
                return 1
        else:
            print("Game Over! Answer: {0}, Your Guess: {1}".format(word, curr))
            return 0