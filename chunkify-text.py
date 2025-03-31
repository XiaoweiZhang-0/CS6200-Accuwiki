def chunkify(data, max_length)-> list:
    """
    Chunkify the text data into smaller segments.

    Args:
        data (list): List of text data to chunkify.
        max_length (int): Maximum length of each chunk.

    Returns:
        list: List of chunked text data.
    """
    chunked_data = []
    for text in data:
        words = text.split()
        chunks = [
            " ".join(words[i : i + max_length])
            for i in range(0, len(words), max_length)
        ]
        chunked_data.extend(chunks)
    return chunked_data