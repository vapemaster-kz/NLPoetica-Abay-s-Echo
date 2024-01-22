import torch
import torch.nn.functional as F

from DataLoader import DataLoader

class MLP:
    def __init__(self, dir_path, context_length):
        self.data_loader = DataLoader(dir_path=dir_path)
        
        self.context_length = context_length

        self.parameters_intialized = False
        self.layers_initialized = False
        
    def possible_characters(self, txts):
        set_characters     = list(set("".join(txts)))
        ordered_characters = sorted(set_characters)

        return ordered_characters

    def initialize_layers(self, emb_dim, hidden_neurons_count=100):
        if not self.parameters_intialized:
            raise ValueError("Initialzie parameters using intialize_parameters()")

        self.C = torch.randn((len(self.stoi), emb_dim))

        self.W1 = torch.randn((self.context_length * emb_dim, hidden_neurons_count))
        self.b1 = torch.randn((hidden_neurons_count, ))

        self.W2 = torch.randn((hidden_neurons_count, len(self.stoi))) * 0.01
        self.b2 = torch.randn((len(self.stoi), )) * 0.01

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        
        # requires grad
        for p in self.parameters:
            p.requires_grad = True
        
        self.layers_initialized = True

    def intialize_parameters(self, add_title=False):
        self.txts = self.data_loader.read_texts_from_folder(add_title=add_title)
        
        unique_characters = self.possible_characters(self.txts)
        self.unique_characters_count = len(unique_characters)

        # lookup tables
        self.stoi = {s:i+1 for i,s in enumerate(unique_characters)}
        self.stoi["<S>"] = 0
        self.stoi["<E>"] = len(self.stoi)
        self.itos = {i:s for s,i in self.stoi.items()}
        
        self.parameters_intialized = True
        self.X, self.Y = self.prepare_data()

    def prepare_data(self):
        if not self.parameters_intialized:
            raise ValueError("Initialzie parameters using intialize_parameters()")

        X, Y = [], []

        for txt in self.txts:
            chrs = list(txt) + ["<E>"]
            context_window = [0] * self.context_length
            for chr in chrs:
                ix = self.stoi[chr]
                X.append(context_window)
                Y.append(ix)
                context_window = context_window[1:] + [ix]
        
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    def forward(self):
        if not self.parameters_intialized:
            raise ValueError("Initialzie parameters using intialize_parameters()")
        
        if not self.layers_initialized:
            raise ValueError("Initialzie weights using initialize_layers()")

        
        # calculating embeddings
        emb = self.C[self.X]

        # changing vector composition
        emb = emb.view(-1, self.context_length*5)

        h = torch.tanh(emb@self.W1 + self.b1)

        logits = h@self.W2 + self.b2

        return logits

    def calculate_loss(self, logits, Y):
        loss = F.cross_entropy(logits, Y)
        return loss

    def pohui_train(self, epochs, lr, verbose=True):
        if not self.parameters_intialized:
            raise ValueError("Initialzie parameters using intialize_parameters()")
        
        if not self.layers_initialized:
            raise ValueError("Initialzie weights using initialize_layers()")
        
        for epoch in range(epochs):
            logits = self.forward()
            loss = self.calculate_loss(logits, self.Y)

            for p in self.parameters:
                p.grad = None
            
            loss.backward()

            for p in self.parameters:
                p.data += -lr * p.grad
            
            if epoch % 100 == 0:
                print(f"Model's loss at {epoch}: {loss}")
    
    @torch.no_grad
    def create_poem(self, max_character = 500):
        output = ""
        local_context_window = [0] * self.context_length

        while True:
            emb = self.C[local_context_window]

            # changing vector composition
            emb = emb.view(-1, self.context_length*5)

            h = torch.tanh(emb@self.W1 + self.b1)

            logits = h@self.W2 + self.b2
            
            probs = F.softmax(logits)

            ix = torch.multinomial(probs, len(probs)).item()
            local_context_window = local_context_window[1:] + [ix]
            output += self.itos[ix]

            if len(output) > max_character:
                break
            
            if ix == len(self.stoi):
                break
        
        return output