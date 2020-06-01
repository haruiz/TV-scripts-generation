# TV Script Generation

In this project, you'll generate your own Seinfeld TV scripts using RNNs. You'll be using part of the Seinfeld dataset of scripts from 9 seasons. The Neural Network you'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

# Example generated script

jerry: what about me?

jerry: i don't have to wait.

kramer:(to the sales table)

elaine:(to jerry) hey, look at this, i'm a good doctor.

newman:(to elaine) you think i have no idea of this...

elaine: oh, you better take the phone, and he was a little nervous.

kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.

jerry: oh, yeah. i don't even know, i know.

jerry:(to the phone) oh, i know.

kramer:(laughing) you know...(to jerry) you don't know.

# LSTM model

```python
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.2):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # define embedding layer which is the responsible for looking at the input integers and basically creating a lookup table
        # input size, output_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        ## Define the LSTM layer
        # input size, output size
        if n_layers > 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,  batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout,  batch_first=True)

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_size)



    # the inpute for each forward is the examples and the hidden state
    def forward(self, nn_input, hidden):
        # extract batch size for the current input
        batch_size = nn_input.size(0)
        # transform current batch size into the embedding representations
        # using word2vec created before
        embeds = self.embedding(nn_input)
        # pass the embedding representation to the model
        lstm_out, hidden = self.lstm(embeds, hidden)
        # flatten the lstm layer(s) outputs for passing into the fully connected layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)        
        # fully-connected layer
        out = self.fc(lstm_out)
        # reshape data into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch,and pass it into the next forward step
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        # initialized the memory of the LSTM layers, an LSTM has a hidden and a cell state that are saved as a tuple
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
```

# Optimization hyperparemeters strategy

 A list with all the hyperparameters involved in the model design, along with a set of possible values for each, was created. Sequently I used Optuna, which is a hyperparameters optimization python library, to automate the testing of them.

For using optuna we start defining a target function, specifying the number of experiments we want to run, then Optuna uses the techniques we chose to search the best hyperparameters values and update them through the trials, always leading to reduce the loss value.

## What is Optuna?
An open source hyperparameter optimization framework to automate hyperparameter search

- Automated search for optimal hyperparameters using Python conditionals, loops, and syntax

- Efficiently search large spaces and prune unpromising trials for faster results

- Parallelize hyperparameter searches over multiple threads or processes without modifying code

## Best hyperparameters:

- sequence_length = 8
- batch_size = 150
- num_epochs = 20
- learning_rate = 0.001
- embedding_dim = 200
- hidden_dim = 250
- n_layers = 1
- Objective Function

## Objective Function

```python

def objective(trial: optuna.Trial):
    # training parameters
    num_epochs = trial.suggest_int("num_epochs", 5, 20)
    learning_rate = trial.suggest_categorical('learning_rate',[0.1, 0.01, 0.001])
    #RNN hyperparameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = trial.suggest_categorical('embedding_dim',[300, 400, 500])
    # Hidden Dimension
    hidden_dim = trial.suggest_categorical('hidden_dim',[150, 250, 350])
    # Number of RNN Layers
    n_layers = trial.suggest_int("n_layers", 1,3)
    #dataset parameters
    sequence_length = trial.suggest_int("num_epochs", 10, 50)  # of words in a sequence
    # Batch Size
    batch_size = 150
    # data loader - do not change
    train_loader = batch_data(int_text, sequence_length, batch_size)

    # define model
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    if train_on_gpu:
        rnn.cuda()
    # defining loss and optimization functions for training    
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    batch_losses = []

    # train model
    rnn.train()
    loss = np.inf
    print("Training for %d epoch(s)..." % num_epochs)
    for epoch_i in range(1, num_epochs + 1):
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)
            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, num_epochs, np.average(batch_losses)))
                batch_losses = []
    return loss
study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=20, direction='minimize')

```
