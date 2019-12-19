# This module hard-codes an RSA model with 1 level of recursion.
# The code is broken down by listeners and speakers, and by individual message/referent, to illustrate the recursive nature
# In practice, this can be implemented much more efficiently with operations on a single numpy array.

import numpy as np


def speaker_two(ref_index, vocabulary, context):
    p_listener_correct = [
        listener_one(message, vocabulary, context)[ref_index]
        for message in vocabulary
    ]
    p_message = p_listener_correct / np.linalg.norm(p_listener_correct, ord=1)
    return p_message


def listener_one(message_ind, vocabulary, context):
    p_speaker_message = [
        speaker_one(ref_index, vocabulary, context)[message_ind]
        for ref_index in range(len(context))
    ]
    p_object = p_speaker_message / np.linalg.norm(p_speaker_message, ord=1)
    return p_object


def speaker_one(ref_index, vocabulary, context):
    # referent is the index corresponding to the referent in C
    p_listener_correct = [
        listener_zero(message_ind, vocabulary, context)[ref_index]
        for message_ind in range(len(vocabulary))
    ]
    p_message = p_listener_correct / np.linalg.norm(p_listener_correct, ord=1)
    return p_message


def listener_zero(message_ind, vocabulary, context):
    """A literal listener."""
    p_object = vocabulary[message_ind]
    p_guess = p_object / np.linalg.norm(p_object, ord=1)
    return p_guess
    # return np.random.choice(context, p=normed)


def speaker_n(ref_index, vocabulary, context, listener_model=listener_zero):
    listener_accuracy = [
        listener_model(message_index, vocabulary, context)[ref_index]
        for message_index in range(len(vocabulary))
    ]
    message_probabilities = listener_accuracy / np.linalg.norm(
        listener_accuracy, ord=1)
    return message_probabilities


def listener_n(message_index, vocabulary, context, speaker_model=speaker_one):
    speaker_meaning = [
        speaker_model(ref_index, vocabulary, context)[message_index]
        for ref_index in range(len(context))
    ]
    object_probabilities = speaker_meaning / np.linalg.norm(
        speaker_meaning, ord=1)
    return object_probabilities


#
# def reference_game(context, vocabulary):
#     # choose referent in context
#     ref = np.random.choice(context)
#     # speaker
#     message = speaker(ref, context, vocabulary)
#     # listener
#     guess = listener(message, context, vocabulary)
#     return guess == ref


def main():
    # Imagine we have three faces: a, b, and c.
    C = ['a', 'b', 'c']  #representation of the context

    # The vocabulary is three words: face, glasses, and hat.
    vocab = [[1, 1, 1], [0, 1, 1], [0, 0,
                                    1]]  #representation of the vocabulary
    # meaning = {"face": ('a', 'b', 'c'), "glasses": ('c', 'b'), "hat": ('c')}

    l0 = np.array([
        listener_zero(0, vocab, C),
        listener_zero(1, vocab, C),
        listener_zero(2, vocab, C)
    ])
    # Right now, all the processing for the whole array is hard-coded in here
    print(l0)

    s1 = np.array([
        speaker_one(0, vocab, C),
        speaker_one(1, vocab, C),
        speaker_one(2, vocab, C)
    ])

    print(s1.T)  #print the transpose to keep rows and columns the same

    l1 = np.array([
        listener_one(0, vocab, C),
        listener_one(1, vocab, C),
        listener_one(2, vocab, C)
    ])
    print(l1)

    s1 = np.array([
        speaker_n(0, vocab, C, listener_zero),
        speaker_n(1, vocab, C, listener_zero),
        speaker_n(2, vocab, C, listener_zero)
    ])

    print(s1.T)  #print the transpose to keep rows and columns the same

    l1 = np.array([
        listener_n(0, vocab, C, speaker_n),
        listener_n(1, vocab, C, speaker_n),
        listener_n(2, vocab, C, speaker_n)
    ])
    print(l1)


main()
