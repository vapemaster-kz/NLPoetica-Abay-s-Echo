import torch.nn.functional as F
import torch

class SimpleRnn():
    def __init__(self, unique_characters, hidden_size):
        self.hidden_size = hidden_size
        possible_characters_count = len(unique_characters) + 2

        self.Wxh = torch.randn((possible_characters_count, hidden_size))*0.001
        self.Whh = torch.randn((hidden_size, hidden_size))*0.001
        self.Why = torch.randn((hidden_size, possible_characters_count))*0.001

        self.b_xh = torch.randn((1, hidden_size))*0.001
        self.b_hy = torch.randn((1, possible_characters_count))*0.001

        self.parameters = [self.Wxh, self.Whh, self.Why, self.b_hy , self.b_xh]
        
        self.stoi = {s:i+1 for i,s in enumerate(unique_characters)}
        self.stoi["<S>"] = 0
        self.stoi["<E>"] = len(self.stoi)

        self.itos = {i:s for s,i in self.stoi.items()}

        for p in self.parameters:
            p.requires_grad = True
                
        self.x_done = False
        self.y_done = False

        self.alphabet = ["<S>"] + unique_characters + ["<E>"]
        
    def string_vectorizer(self, strng, alphabet):
        vector = [0 if char != strng else 1 for char in alphabet]
        return torch.tensor(vector, dtype=torch.float)
    
    def forward(self, emb, h_t):
        h_t = F.relu(emb @ self.Wxh + self.b_xh + h_t @ self.Whh)

        logits = (h_t @ self.Why + self.b_hy)

        return logits, h_t
    
    def prepare_data(self, x):
        if self.x_done and self.y_done:
            return
        
        self.X, self.Y = [[]], [[]]

        # self.X ==> [num_sentences, num_characters, num_characters_alphabet]
        idx = 0

        for x_idx, sent in enumerate(x):
            chrs = ["<S>"] + list(sent) + ["<E>"]
            
            for ch1, ch2 in zip(chrs, chrs[1:]):
                ix = self.string_vectorizer(ch1, self.alphabet)
                iy = self.stoi[ch2]
                
                self.X[idx].append(ix)
                self.Y[idx].append(iy)
            
            if (x_idx != len(x)-1):
                self.X.append([])
                self.Y.append([])
                idx += 1

        self.x_done = True
        self.y_done = True

    def create_batch(self, x, y, batch_size):
        assert len(x) == len(y)

        indices = torch.randint(0, len(x), (batch_size,))
        batched_x = []
        batched_y = []

        for i in range(batch_size):
            batched_x.append(x[indices[i]])
            batched_y.append(y[indices[i]])
        
        return batched_x, batched_y

    def train(self, epochs, dataset, lr=0.05):
        self.prepare_data(dataset)
        
        batch_sentences_count = int(len(dataset)*0.25)

        for epoch in range(epochs):
            batched_x, batched_y = self.create_batch(self.X, self.Y, batch_sentences_count)
            total_loss = 0.0
            num_sentences = 0

            for idx, sentence in enumerate(batched_x):
                prev_h_t = torch.zeros((1, self.hidden_size))

                for chr_idx, chr in enumerate(sentence):
                    # Zero out gradients
                    for p in self.parameters:
                        p.grad = None

                    # Forward pass
                    logits, h_t = self.forward(chr, prev_h_t)

                    # Calculate loss
                    y_true = batched_y[idx][chr_idx]
                    loss = F.cross_entropy(logits[0], torch.tensor(y_true))
                    total_loss += loss.item()
                    
                    h_t_detached = h_t.detach()
                    num_sentences += 1

                    # Backward pass
                    loss.backward()

                    # clipping exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)  # Adjust max_norm as needed
                    
                    # Update parameters
                    for p in self.parameters:
                        p.data += -lr*p.grad

                    prev_h_t = h_t_detached
                    
            avg_loss = total_loss / num_sentences
            print(f"Batch {epoch+1}, Average Loss: {avg_loss}")

    @torch.no_grad
    def create_poem(self, max_character = 500):
        output = ""
        ix = self.string_vectorizer("<S>", self.alphabet)
        prev_h_t = torch.zeros((1, self.hidden_size))

        while (True):
            logits, h_t = self.forward(ix, prev_h_t)
            probs = torch.softmax(logits, dim=-1)
            multinomial_val = torch.multinomial(probs[0], 1).item()
            prev_h_t = h_t
            
            chr = self.itos[multinomial_val]
            arr = torch.zeros((len(self.alphabet),))
            arr[multinomial_val] = 1

            ix = arr.clone()

            if chr == "<E>":
                break

            output += chr
                
            if len(output) > max_character:
                break
        
        return output        