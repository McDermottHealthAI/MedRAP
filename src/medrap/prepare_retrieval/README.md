# medrap.prepare_retrieval

Offline preparation for the `medrap prepare-retrieval-dataset` command. Takes a Hugging Face
dataset of text documents, renders them into a fixed field order, tokenizes and embeds them
using a pretrained encoder, and writes out a FAISS index for nearest-neighbor retrieval at
training and inference time.
