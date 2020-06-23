# Bhojpuri_Chunking_Module
Implemented Deep Learning Module for Chunking Process of Bhojpuri Language.
# Installation
Install Keras, Keras_contrib and Keras Self Attention module.
# Files
chunking.py:- training module for chunking process

conlleval.py:- evaluator for chunking

predict.py:- Takes sentences along with POS tags and returns the IOB tags corresponding to them.

input.txt:- Input for predict.py

output.txt:- Output generated for reference

model.h & model.json :- Saved model weights for BiLSTM + CRF without Self-Attention.

model(attention).h & model(attention.json) :- Saved model weights for BiLSTM + CRF with Self-Attention.

vocab.txt :- Saved vocabulary for pos_tag and word.

ssfconll.py:- Converts ssf to the format in which model chunking.py takes input along with IOB_tags for training.

bhojpuri_chunked_clean.txt:- SSF format input for chunking.

Chunk_file_temp_iob.txt:- Formatted input genearated from ssfconll.py
