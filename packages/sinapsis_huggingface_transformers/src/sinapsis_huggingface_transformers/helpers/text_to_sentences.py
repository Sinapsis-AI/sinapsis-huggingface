# -*- coding: utf-8 -*-
import re


def sentences_to_n_words(input_text: str, n_words: int) -> list[str]:
    """Split an input text into sentences of n_words.

    Args:
        input_text (str): input text to be split into sentences.
        n_words (int): number of words for each sentence.

    Returns:
        list[str]: list of sentences having n_words (or less in case of the final sentence).
    """
    words_text = input_text.split(" ")
    sentences = []
    if n_words >= len(words_text):
        sentences.append(input_text)
    else:
        for i in range(0, len(words_text), n_words):
            sentence = " ".join(words_text[i : i + n_words])
            sentences.append(sentence)
    return sentences


def split_text_into_sentences(input_txt: str, delimiters: str = r"[.,;?!]") -> list[str]:
    """Split the text into sentences using the provided delimiters (the delimiters are kept).

    Args:
        input_txt (str): input text to be split into sentences.
        delimiters (str, optional): regular expression with delimiters. Defaults to r"[.,;?!]".

    Returns:
        list[str]: list of sentences.
    """
    splitted_text = re.split(f"({delimiters})", input_txt)

    # Combine each sentence with its delimiter (except the last segment):
    sentences = [splitted_text[i] + splitted_text[i + 1] for i in range(0, len(splitted_text) - 1, 2)]

    # If there's a remaining string (the last one), add it as is:
    if len(splitted_text) % 2 == 1:
        sentences.append(splitted_text[-1])

    # Remove any empty strings or blank spaces:
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences
