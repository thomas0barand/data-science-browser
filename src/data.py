import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


def load_queries(file_path: str) -> Dict[str, Dict]:
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            queries[data['_id']] = data
    return queries


def load_corpus(file_path: str) -> Dict[str, Dict]:
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            corpus[data['_id']] = data
    return corpus


def load_valid_tsv(file_path: str) -> List[Tuple[str, str, int]]:
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                query_id, corpus_id, label = parts
                pairs.append((query_id, corpus_id, int(label)))
    return pairs


class IRTrainDataset(Dataset):
    def __init__(self, queries_path: str, corpus_path: str, valid_path: str):
        print("Loading queries...")
        self.queries = load_queries(queries_path)
        print(f"Loaded {len(self.queries)} queries")
        
        print("Loading corpus...")
        self.corpus = load_corpus(corpus_path)
        print(f"Loaded {len(self.corpus)} corpus documents")
        
        print("Loading valid pairs...")
        all_pairs = load_valid_tsv(valid_path)
        
        # Filter pairs where both query and corpus exist
        self.pairs = []
        skipped = 0
        for query_id, corpus_id, label in all_pairs:
            if query_id in self.queries and corpus_id in self.corpus:
                self.pairs.append((query_id, corpus_id, label))
            else:
                skipped += 1
        
        print(f"Loaded {len(self.pairs)} valid pairs ({skipped} skipped)")
        
        # Get unique train query IDs
        self.train_query_ids = set(pair[0] for pair in self.pairs)
        print(f"Train set contains {len(self.train_query_ids)} unique queries")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query_id, corpus_id, label = self.pairs[idx]
        query = self.queries[query_id]
        candidate = self.corpus[corpus_id]
        
        return {
            'query_id': query_id,
            'query_title': query.get('text', ''),
            'query_metadata': query.get('metadata', {}),
            'corpus_id': corpus_id,
            'corpus_title': candidate.get('title', candidate.get('text', '')),
            'corpus_text': candidate['text'],
            'corpus_metadata': candidate.get('metadata', {}),
            'label': label
        }


class IRTestDataset(Dataset):
    def __init__(self, queries_path: str, corpus_path: str, valid_path: str):
        print("Loading queries...")
        all_queries = load_queries(queries_path)
        print(f"Loaded {len(all_queries)} total queries")
        
        print("Loading corpus...")
        self.corpus = load_corpus(corpus_path)
        print(f"Loaded {len(self.corpus)} corpus documents")
        
        print("Loading valid pairs to identify train queries...")
        train_pairs = load_valid_tsv(valid_path)
        train_query_ids = set(pair[0] for pair in train_pairs)
        print(f"Identified {len(train_query_ids)} train query IDs")
        
        # Test queries are those NOT in valid.tsv
        self.test_queries = {qid: q for qid, q in all_queries.items() 
                            if qid not in train_query_ids}
        self.query_ids = list(self.test_queries.keys())
        
        print(f"Test set contains {len(self.test_queries)} queries")
    
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query = self.test_queries[query_id]
        
        return {
            'query_id': query_id,
            'query_title': query.get('text', ''),
            'query_metadata': query.get('metadata', {})
        }


def collate_ir_batch(batch):
    if not batch:
        return {}
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key == 'label':
            collated[key] = torch.tensor(values, dtype=torch.long)
        else:
            collated[key] = values
    return collated


def get_train_loader(queries_path='data/queries.jsonl', 
                     corpus_path='data/corpus.jsonl',
                     valid_path='data/valid.tsv',
                     batch_size=32, 
                     shuffle=True):
    dataset = IRTrainDataset(queries_path, corpus_path, valid_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ir_batch,
    )


def get_test_loader(queries_path='data/queries.jsonl',
                    corpus_path='data/corpus.jsonl',
                    valid_path='data/valid.tsv',
                    batch_size=32):
    dataset = IRTestDataset(queries_path, corpus_path, valid_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ir_batch,
    )


if __name__ == "__main__":
    train_loader = get_train_loader(batch_size=8, shuffle=True)

    second_batch_iter = iter(train_loader)
    next(second_batch_iter)  # Skip the first batch
    second_batch = next(second_batch_iter)

    test_loader = get_test_loader(batch_size=8)
    
    print("\nSecond batch sample:")
    second_batch_iter = iter(test_loader)
    next(second_batch_iter)  # Skip the first batch
    second_batch = next(second_batch_iter)
