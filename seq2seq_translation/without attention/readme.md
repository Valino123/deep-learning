# Data Preprocessing
## class:Lang
1. attach each word with an index, increment from 0
2. create maps from word to index as well as index to word
3. addSentence() and addWord() are used for creating the dict
## func: prepareData(lang1, lang2, reverse=False):
1. pair: pair[0] is the src language, pair[1] is the target language
2. readLangs(): construct pairs, Lang instance for input(src) and output(target) language
3. cass readLangs() to read languages from input
4. construct dict instance for input_lang and output_lang
## func: indicesFromSentence(lang, sentence)
1. Encode input sentence: construct an array with length of the sentence, and replace each word with its index in dict
## func: tensorFromSentence(lang, sentence)
1. call indicesFromSentence to encode the sentence
2. append end-of-sentence token
3. convert this encodings(array) to 2d tensor with element type long int and shape<1, ?>
>**Note:** The view function returns a new tensor with the same data but of a different shape3. In this case, .view(1, -1) is used to reshape the tensor into a 2D tensor with 1 row. The -1 is a placeholder that PyTorch will fill in with the appropriate number to maintain the same number of elements in the tensor4.
## func: tensorsFromPair(pair)
1. call tensorFromSentence to construct input&output tensor
## get_dataloader(batch_size):
1. calls prepareData to create Lang type instance for input language and target language
2. create input_ids and target_ids, which store the encoded sentence containing end-of-sentence token
3. construct train_dataloader with pytorch-functions like TensorDataset, RandomSampler and DataLoader
## structure
get_dataloader
- prepareData
    - readLangs
    - addSentence
        - addWord
- indicesFromSentence
- TensorDataset
- RandomSampler
- DataLoader
# Encoder
## Diagram
![alt text](image.png)
## Process
1. input_tensor: 
- #input_tensor = 114450(numbers of pairs/sentences)
- element type is long
- the sentence is represented by indices of words
- DataSet is composed of input_ids and output_ids here
    - ![alt text](dc0d2e656e1624178d5c81c4641bfd8.png)
    - ![alt text](4d744d1ca03a30f1b4a6e066ffb6e0b.png)
>**Note:** DataLoader: This is a Python iterable over a dataset, provided by PyTorch’s torch.utils.data package. It supports both map-style and iterable-style datasets, automatic batching, single- and multi-process data loading, and automatic memory pinning. It decouples the data loading process from the model training code for better readability and modularity.
>**Note:** RandomSampler: This is a type of sampler in PyTorch that returns random indices. It samples elements from the dataset randomly without replacement. It’s used to specify the sequence of indices/keys used in data loading.
>**Note:** TensorDataset: This is a dataset wrapping tensors. Each sample will be retrieved by indexing tensors along the first dimension. It’s a ready-to-use class in PyTorch to represent your data as a list of tensors.
2. get embeddings
- why?
    - Semantic information: Word-index map can't express the semantic information, like 'I like traveling' and 'I love go hiking' distinguishes in word-index space. But through embeddings, they are alike in the embedding semantic space
    - Variable Length Sequences: Embeddings allow the model to handle variable length sequences
- how?
    - construct self.embedding with expected size before and after embedding:
        ```
        self.embedding = nn.Embedding(input_size, hidden_size)    
        ```
    - introduce dropout for better training
    - Code:
        ```
        embedded = self.dropout(self.embedding(input))
        ```
3. Feed the processed input into GRU(one sort of RNN) model
## Shape
Now let's figure out how the shape of data tensor changes over time
1. dict variables
- #fra_words = 4601, #eng_words = 2991
- #pairs/sentences = 14450
- input_tensor/output_tensor: \[32,10\] ---batch_size(32 sentences), sentence_size(10 words)
2. embedding
- [32, 10, 128] ---batch_size(32 sentences), sentence_size(10 words), embedding_size(128 per word embedding)
- ![alt text](d68585ead6d9287e3eeff75808c337f.png)
3. output
- [32, 10, 128] ---same to embedding
4. hidden
- [ 1, 32, 128] ---#Layer(1, specified by this GRU), batch_size(32), hidden state size(128 for each input sentence)
- ![alt text](9a38f2281eb45e3233f3fee078e3b70.png)
## ONNX model
- ![alt text](1712983385088.png)
# Decoder
## Diagram
![alt text](image-1.png)
## Process
```
decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)
decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
```
1. prepare data for the first decoder unit
- input: construct \[32,1\] matrix with SOS_token
- hidden: the encoder_hidden
2. feed data into GRU word by word, from 3 to 7 
3. get embedding
- construct self.embedding:
    - shape from 2991(output_size, #eng_words) to 128(hidden_size)
    - code:
        ```
        self.embedding = nn.Embedding(output_size, hidden_size)
         output = self.embedding(input)
        ```
4. do relu to the embedding
- Why?
    - to avoid gradient vanishment
- How?
    - The gradient is turned into 0 for negative inputs and 1 for positive inputs
```
output = F.relu(output) 
```
5. feed data into GRU:
```
self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
output, hidden = self.gru(output, hidden)
```
6. linearly transform output from size of 128 to size of 2991
- Why?
    - It's the denser layer/ fully connective layer
    - Inductive Bias: It introduces a form of inductive bias into the model. This makes the method more special purpose and less flexible, but often much more useful(?)
    - Control output size&shape
- How?
    - Weighted Linear Combination of Inputs
    - Learnable Parameters
    - Mapping
    ```
    self.out = nn.Linear(hidden_size, output_size)
    output = self.out(output)
    ```
7. prepare data for next unit
- for training 
    - we have target tensor, so just select the ith word from each sentences in the current batch and combine them into an vector as the next input
    - target_tensor[:,i]: This is indexing into the target_tensor. The colon : means we’re taking all elements along the first dimension (usually representing different data samples in the batch), and i means we’re taking the i-th element along the second dimension (usually representing different features or sequence positions). So, target_tensor[:,i] gives us the i-th feature or sequence position for all data samples.
    - ![alt text](9d710be3392b9ef0008dff1eb2ee4be.png)
    - .unsqueeze(1): This is a call to the unsqueeze function of PyTorch. The unsqueeze function returns a new tensor with a dimension of size one inserted at the specified position. In this case, it’s adding a new dimension at position 1. This is often done to match the expected input shape of certain functions or methods2. For example, if the original shape of target_tensor[:,i] was (batch_size,), after unsqueeze(1), the shape would become (batch_size, 1). This might be necessary because many PyTorch functions expect inputs to have a certain number of dimensions
    - ![alt text](0d88d6a1cb44bdb34e183786dc7db08.png)
- for testing
    - decoder_output is of shape[32, 1, 2991], it actually stores the probability of each word from the eng_dict. The one with the highest probability tends to be the predicted word in the current decoder unit
    - _, topi = decoder_output.topk(1): This line is using the topk function from PyTorch. The topk function returns the k largest elements of the given input tensor along a given dimension. If dim is not given, the last dimension of the input is chosen. In this case, k is 1, so it’s finding the largest element (i.e., the most probable next word in the sequence). The function returns two tensors: the top k values and their corresponding indices. The _ is a common Python idiom for ignoring a value; here, it’s ignoring the actual values and only keeping the indices (topi).
    - decoder_input = topi.squeeze(-1).detach(): This line is preparing the top indices for the next input to the decoder.
        - The squeeze function removes dimensions of size 1 from the tensor. The argument -1 means it removes the last dimension if it’s of size 12. This is often used when you have a single sample and the model expects a batch, so you need to remove the singleton dimension to match the expected input shape.
        - The detach function creates a tensor that shares storage with the original tensor, but with its computation history detached. This means the new tensor will not have gradients computed for it during backpropagation. This is used here because the next input to the decoder should not be updated during backpropagation.
8. get the output
merging all the decoder output tensors (each representing the output at a specific time step) into a single tensor. This is done along the feature dimension, so the resulting tensor (decoder_outputs) has the same number of features as the individual output tensors, but its length (along the feature dimension) is the sum of their lengths
```
decoder_outputs = torch.cat(decoder_outputs, dim=1)
```
9. do softmax
## Shape
Now let's figure out how the shape of data tensor changes over time
1. input parameters:
- encoder_outputs: [32 ,10 ,128]
- encoder_hidden: [1, 32, 128]
- target_tensor: [32, 10]
2. decoder_input: [32, 1] --- batch_size(32), feature_size(1, each time process one word)
3. in forward_step(input, hidden):
- output(embedded, from gru): [32, 1, 128] --- batch_size, feature_size, word_embedding_size
    - ![alt text](97d248dd87018fc0c27a56c8baccc89.png)
- output(after the dense layer self.out): [32, 1, 2991] --- batch_size, feature_size, #classes(#eng_dict = 2991, each element among this 2991 is associated with a probability value like in softmax)
4. results from forward_step
- decoder_output: [32, 1, 2991]
- decoder_hidden: [1, 32, 128]
5. decoder_input:
- test: main idea is to select the most possible word from decoder_output's probability dimension, get their indices and turn them into shape [32, 1] --- batch_size, feature_size
- train: main idea is to select the next word from target_tensor(batch of 32)'s each sentence, them combine them to the next input, also shape [32, 1]
6. return values:
we have a list of 10 tensors (each representing the output of the decoder for a specific time step), and each tensor has a shape of [32, 1, 2991]. When we call torch.cat(decoder_outputs, dim=1), it concatenates these tensors along the second dimension (the one with size 1). The resulting tensor has a shape of [32, 10, 2991], which represents the batch size, the sequence length (number of words per sentence), and the size of the dictionary, respectively
- decoder_outputs: [32, 10, 2991]
## ONNX model
![alt text](decoder.png)
# Train epoch
1. define optimizer: Adam. 
They’re responsible for updating the parameters of the encoder and decoder during training
```
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
```
2. Iterate the dataloader, add the losses up and compute the average loss. Details are as follows
3.  Zero out the gradients of the encoder and decoder
```
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()
```
4. run encoder and decoder
```
encoder_outputs, encoder_hidden = encoder(input_tensor)
decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
```
5. Compute Loss:
The loss is computed between the decoder_outputs and the target_tensor using the criterion. The view method is used to reshape the tensors for the loss computation.
```
criterion = nn.NLLLoss()
```
The nn.NLLLoss() stands for Negative Log-Likelihood Loss. It's commonly used for training a classification problem

```
loss = criterion(
    decoder_outputs.view(-1, decoder_outputs.size(-1)),
    target_tensor.view(-1)
)
```
The view method is used to reshape a tensor without changing its data.
6. Call backward method on the loss to compute gradients
```
loss.backward()
```
7. Call step method on the optimizers to update the parameters of encoder and decoder
```
encoder_optimizer.step()
decoder_optimizer.step()
```
