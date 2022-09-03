# twitterSentiment
A simple example of Bert model used for sentiment analysis.
Dataset source: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
This dataset provides us with product reviews on Twitter and their sentiments('Irrelevant', 'Negative', 'Neutral', 'Positive'). 
Since the purpose of this example is to analyze users' sentiment about products, records with " Irrelevant " labels are removed.

To solve this probelm, I used bert model in two different ways:

+ To tokenize the raw input, I used Wordpiece algorithm. This tokenizer is trained on the whole dataset. The model is created by using a small Bert and a hidden layer on top of the Bert layer. The final layer receives the pooled_ouput of the Bert layer and generates the probability of each label by using the softmax activation function.
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 2,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.21.2",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
    
+ the second way is to use pre_trained model, in this example, I chose twitter-roberta-base-sentiment-latest. This model is trained based on ~124M tweets.
More info at: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
