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


#Encode amino acid sequences for input into model and split test and training data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.20, shuffle=True)

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
num_amino_acids = 20
max_length = max(len(s) for s in sequences)

def one_hot_encode_sequence(seq, aa_to_int, max_length):
    one_hot_encoded = np.zeros((max_length, num_amino_acids), dtype=int)
    for i, aa in enumerate(seq[:max_length]):
        if aa in aa_to_int:
            one_hot_encoded[i, aa_to_int[aa]] = 1
    return one_hot_encoded

train_encoded_sequences = [one_hot_encode_sequence(sequence, aa_to_int, max_length) for sequence in train_sequences]
test_encoded_sequences = [one_hot_encode_sequence(sequence, aa_to_int, max_length) for sequence in test_sequences]
train_array = np.array(train_encoded_sequences)
test_array = np.array(test_encoded_sequences)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


#Create train and test datasets and model
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


train_dataset = ProteinDataset(train_array, train_labels)
test_dataset = ProteinDataset(test_array, test_labels)

class ProteinModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


input_dim = 500 * 20
hidden_dim = 128
output_dim = 2

model = ProteinModel(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
