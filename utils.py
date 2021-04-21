from typing import List


def get_reciprocal_rank(retrieved_docs: List, relevant_docs: List) -> float:
    """
    The mean reciprocal rank is a statistic measure for evaluating any process that produces a list of possible
    responses to a sample of queries, ordered by probability of correctness.

    rank_i: The reciprocal rank of a query response is the multiplicative inverse of the rank of the first correct answer-

    :param retrieved_docs: List of queries and their retrieved documents (from evaluated system)
    :param relevant_docs:  List of queries and their relevant documents (from gold standard)
    :return:
    """

    for rank, retrieved_doc in enumerate(retrieved_docs, start=1):
        if retrieved_doc in relevant_docs:
            return 1. / rank

    return 0.


def get_avg_precision(retrieved_docs: List, relevant_docs: List) -> float:
    retrieved_relevant_docs = 0.
    precision_sum = 0.

    # Compute avg. precision
    if len(relevant_docs) > 0:
        for rank, retrieved_doc in enumerate(retrieved_docs, start=1):
            if retrieved_doc in relevant_docs:
                retrieved_relevant_docs += 1
                precision_sum += retrieved_relevant_docs / rank

        return precision_sum / len(relevant_docs)
    else:
        return 0.


def get_mean_avg_precision(queries_retrieved_docs: List[List], queries_relevant_docs: List[List]) -> float:
    """

    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    :param queries_retrieved_docs: List of queries and their retrieved documents (from evaluated system)
    :param queries_relevant_docs: List of queries and their relevant documents (from gold standard)
    :return: MAP score
    """

    assert len(queries_retrieved_docs) == len(queries_relevant_docs)

    sum_avg_precision = 0.

    # Iterate over all queries
    for query_idx, retrieved_docs in enumerate(queries_retrieved_docs):
        relevant_docs = queries_relevant_docs[query_idx]

        sum_avg_precision += get_avg_precision(retrieved_docs, relevant_docs)

    return sum_avg_precision / len(queries_retrieved_docs)


def highlight_max(data, color='green'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.replace('%','', regex=True).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)