### NLP-project
# Metric for human-chatbot dialogue quality
Anastasiya Karpovich  
Moscow Institute of  Physics and Technology  
nascar.by@gmail.com  

### Abstract
Chatbot is a program that can chat with a user on a free topic. Evaluation of  the quality of the chatbot is very difficult. The main problem is that there is no "right" answer in the dialogue task, with which one can compare what the system has given out. Therefore, the metrics that are used to determine the quality of machine translation are not appropriate in this case. In this case, a person can easily evaluate the quality of the bot: it is enough to read the system's answer to understand whether it is fit into the dialogue. The dataset  of 4224 human-bot and 526 human-human dialogues was assembled during the Conversational Intelligence Challenge, each dialogue is estimated by a person on a five-point scale.  In this paper we describe  neural network that is taught on this  dataset to evaluate chatbot dialogues without the participation of a person. All code used here is available online at: https://github.com/nascarr/NLP_project.

### Data collection and dataset describtion
The dataset contains  4,750 dialogues. These include 4,224 human-to-bot dialogues and 526 human-to-human conversations. 
The dataset was collected during  the Conversational Intelligence Challenge in 2017 (Logacheva et al., 2018). 
A chatbot and a human volunteer should discuss an excerpt from a Wikipedia article that was provided. These texts were taken from the SQuAD dataset (Rajpurkar et al., 2016).
The peers were encouraged (but not strictly required) to discuss this text.
The discussion occurred through framework for collecting conversations of humans and bots
which operated on Telegram and Facebook messaging service. When a user
started a conversation, the framework randomly assigned her a bot or another
user, so the user did not know if s/he was talking to a bot or a human.
After the conversation was finished, user was asked to evaluate the whole dialogue along three dimensions: overall quality of dialogue, breadth of dialogue
and
engagement of peer. All three parameter were given scores from 1 to 5. We use only “quality” mark for neural network training.

### Data preprocessing
Duplicates (43 threads), 1747 short threads and 144  threads with long utterances were deleted from the dataset. We consider thread as short if it includes less than 2 utterances for at least one user. And we consider utterance as long if it consists of more than 200 words. For training baseline solution we use all human-bots threads one time and human-human threads two times: each time with different mark. For training HRE solution also we use user ids for each utterance in thread. Final number of threads in our dataset  is 3154.
Before neural network training data was randomly split into two subsets: for training and validation. The size of validation set was 10\% of all data. We split dataset before duplicating human-human threads. We use 5 different fixed random states for splitting to compare models.

### Baseline solution
Baseline solution is one layer RNN with GRU or bidirectional GRU cell with linear classifier on the top. We use 100-dimensional word embeddings pre-trained with GloVe (Pennington et al., 2014) on
Wikipedia and Gigaword corpora. Dialogue is concatenated like regular text with special token 'EOU' at the end of each utterance but without any user names and ids. Embedding of token 'EOU' is just vector of zeros.  Learning rate was set to 0.0003 for Adam optimizer. Model was trained for 60 epochs while minimizing mean square error for predicted scores. Results of training are shown in Table 1 (average of 5 runs). For each RNN modification we present minimal validation loss, maximal pearson and spearman correlations for validation set that were observed during training. 

#### Table 1. Baseline solution. 
| RNN description | Validation loss | Pearson | Spearman |
|---|---|---|---|
|GRU, 100 units, 1 layer | 1.64 | 0.39 | 0.39|
|BiGRU, 100 units, 1 layer |  1.66| 0.37 | 0.37|

### Hierarchical RNN encoder
Hierarchical RNN encoder (HRE) is used as main architecture (Lowe et al., 2016).  It consists of two  RNNs: the first takes utterances as inputs and returns phrase embeddings. The second RNN takes phrase embeddings as inputs and returns dialogue embeddings. Both RNNs are GRU cells and consist of 1 layer of 100 units. Last layer is dense linear layer with one neuron. User ids we attach to phrase embeddings as [1,0] and [0,1] vectors. HRE diagram is shown at Figure 1.
We used Adam optimizer and learning rate was 0.0003. Model was trained for 60 epochs while minimizing mean square error as loss. Results of training  are shown in table 2 (average of 5 runs).  
So, best pearson correlation between predicted marks and human evaluations that we 
observed is 0.42.


#### Table 2. HRE solution.
|RNN description | Validation loss |Pearson | Spearman|
|---|---|---|---|
|1st RNN - GRU, 2nd RNN - GRU, 100 units, 1 layer |  1.6 | 0.42  | 0.42|


#### Figure 1. HRE solution diagram.
![alt text](https://github.com/nascarr/NLP_project/blob/master/HRE_diagram.PNG)

### Acknowledgements
I am grateful to Varvara Logacheva for mindful mentoring and Valentin Malykh for tough but great course "Deep Learning in Natural Language Processing".

### References
1. [Logacheva et al., 2018] Logacheva et al.,2018. A Dataset of Topic-Oriented Human-to-Chatbot
Dialogues. https://github.com/DeepPavlov/convai/blob/master/2017/data/dataset_description.pdf.
2. [Lowe at al., 2016] Lowe, R., Serban, I. V., Noseworthy, M., Charlin, L., and Pineau, J., 2016. On the Evaluation of
Dialogue Systems with Next Utterance Classification. 
3. [Pennington et al., 2014] Pennington, J., Socher, R., and Manning, C. D., 2014. Glove: Global vectors for word representation. Empirical Methods in Natural Language Processing.
4. [Rajpurkar et al., 2016] Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P., 2016. SQuAD: 100,000+ Questions for Machine
Comprehension of Text.
