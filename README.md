# hangman-transformer
Using transformers to solve the Hangman Game

![Hangman](https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Hangman_example.svg/440px-Hangman_example.svg.png)

[The Hangman Game](https://en.wikipedia.org/wiki/Hangman_(game)): Guess a word letter by letter while making no more than 6 wrong guesses. Correctly guessing a letter reveals the position of the letter in the word.

The model predicts the most probable letter present in the word given the masked word.
Example, given "TRANSFO_ME_" the correct guess should be "R"

File description:
1. generate_data.ipynb: Generate training/testing dataset from list of words.
2. hangman.py: Class to simulates hangman games and bechmark model.
3. model.py: Classes defining model artitechtue.
4. solver.py: Class defining a hangman solver built on top of the model.
5. train.ipynb: Notebook to train the model.
6. play.ipynb: Notebook to benchmark the model by simulating hangman games.