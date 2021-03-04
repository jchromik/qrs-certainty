def rescale(index_pairs, old_window_size, new_window_size):
    return [
        (chunk_index, (window_index*new_window_size) // old_window_size)
        for chunk_index, window_index in index_pairs]


def index_pair(window_index, window_size, chunk_sizes):
    if window_index < 0:
        raise IndexError("Window index negative.")

    for chunk_index, chunk_size in enumerate(chunk_sizes):
        usable_chunk_length = chunk_size - window_size
        if window_index <= usable_chunk_length:
            return chunk_index, window_index
        else:
            window_index -= (usable_chunk_length + 1)

    raise IndexError("Window index out of bounds.")


def indexes_for_batch(batch_index, batch_size):
    start = batch_index * batch_size
    end = (batch_index+1) * batch_size
    return range(start, end)


def index_pairs_for_batch(batch_index, batch_size, window_size, chunk_sizes):
    return [
        index_pair(window_index, window_size, chunk_sizes)
        for window_index in indexes_for_batch(batch_index, batch_size)]
