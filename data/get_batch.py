
def get_batch(source, i, bqtt_len):
    seq_len = min(bqtt_len, len(source)-i-1)
    data = source[i:i+seq_len]
    target = source[i+1:i+seq_len+1]
    return data, target
