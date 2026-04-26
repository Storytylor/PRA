import numpy as np

states = ["E", "I"]
symbols = ["A", "C", "G", "T"]

def load_data(file):
    sequences = []
    with open(file) as f:
        for line in f:
            if line.strip():
                sequences.append(line.strip().split(",")[-1])
    return sequences

def train(sequences):
    trans = np.ones((2, 2))
    emit = np.ones((2, 4))
    init = np.ones(2)

    for seq in sequences:
        split = len(seq) // 2
        hidden = [0]*split + [1]*(len(seq)-split)

        init[hidden[0]] += 1

        for i, ch in enumerate(seq):
            if ch in symbols:
                emit[hidden[i], symbols.index(ch)] += 1
            if i > 0:
                trans[hidden[i-1], hidden[i]] += 1

    trans /= trans.sum(axis=1, keepdims=True)
    emit /= emit.sum(axis=1, keepdims=True)
    init /= init.sum()

    return trans, emit, init

def viterbi(seq, trans, emit, init):
    n, T = 2, len(seq)
    dp = np.zeros((n, T))
    path = np.zeros((n, T), int)

    for s in range(n):
        dp[s, 0] = init[s] * emit[s, symbols.index(seq[0])]

    for t in range(1, T):
        for s in range(n):
            probs = dp[:, t-1] * trans[:, s] * emit[s, symbols.index(seq[t])]
            path[s, t] = np.argmax(probs)
            dp[s, t] = np.max(probs)

    states_seq = [np.argmax(dp[:, -1])]
    for t in range(T-1, 0, -1):
        states_seq.append(path[states_seq[-1], t])
    
    return "".join(states[s] for s in states_seq[::-1])

# Run
seqs = load_data("dna.csv")
trans, emit, init = train(seqs)

print(viterbi(seqs[0], trans, emit, init))
