def pad(sequences):
    lengths = [len(x) for x in sequences]
    max_len = max(lengths)
    trailing_dims = sequences[0].size()[1:]
    dims = (max_len, len(sequences)) + trailing_dims
    padded = sequences[0].new(*dims).zero_()
    for i, seq in enumerate(sequences):
        padded[:lengths[i], i] = seq
    return padded, lengths

def collate_fn(batch):
    """Dynamic padding of batch"""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    inputs, targets = zip(*batch)
    inputs, inputs_lengths = pad(inputs)
    targets, _ = pad(targets)
    return inputs, inputs_lengths, targets
