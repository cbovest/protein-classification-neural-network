import torch
from torch import nn
import torch.optim as optim
import requests
import pandas
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Cft_transmem&format=tsv&query=%28%28organism_id%3A9606%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B80+TO+500%5D%29%29"
uniprot_request = requests.get(query_url)

bio = BytesIO(uniprot_request.content)

df = pandas.read_csv(bio, compression='gzip', sep='\t')


#Set all NA-values in Transmembrane to false. Create 2 dataframes, one for transmembrane
#proteins and another for non-transmembrane proteins
df['Transmembrane'] = df['Transmembrane'].fillna("FALSE")

transmembrane = df['Transmembrane'].str.contains("TRANSMEM")
non_transmembrane = df['Transmembrane'].str.contains("FALSE")

transmembrane_df = df[transmembrane & ~non_transmembrane]
non_transmembrane_df = df[non_transmembrane & ~transmembrane]

#Create a list of sequences for each dataframe from previous step, and a corresponding
#list of labels for each sequence(1 for transmembrane proteins, 0 for non-transmembrane)
transmembrane_seqs = transmembrane_df['Sequence'].tolist()
transmembrane_labels = [1 for protein in transmembrane_seqs]
non_transmembrane_seqs = non_transmembrane_df['Sequence'].tolist()
non_transmembrane_labels = [0 for protein in non_transmembrane_seqs]

sequences = transmembrane_seqs + non_transmembrane_seqs
labels = transmembrane_labels + non_transmembrane_labels
len(sequences) == len(labels)
