from typing import Optional, List, Union

import dspy
from dsp import dotdict
from dspy import Prediction
from rag4p.rag.store.local.internal_content_store import InternalContentStore


class LocalRM(dspy.Retrieve):

    def __init__(self, content_store: InternalContentStore, k: int = 3):
        self.content_store = content_store
        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> Prediction:
        k = k if k is not None else self.k
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]
        passages = []

        for query in queries:
            results = self.content_store.find_relevant_chunks(query, k)

            for index, chunk in enumerate(results):
                passages.append(dotdict({"long_text": chunk.chunk_text, "index": index}))

        # return dspy.Prediction(
        #     passages=passages,
        # )

        return passages
