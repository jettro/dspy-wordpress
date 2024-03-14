from typing import Optional, Union, List

import dspy
from dsp import dotdict
from dspy import Prediction
from rag4p.rag.embedding.embedder import Embedder
from rockset import RocksetClient
from rockset.model.query_parameter import QueryParameter


class RocksetRM(dspy.Retrieve):

    def __init__(self,
                 rockset_workspace_name: str,
                 rockset_client: RocksetClient,
                 query_lambda_name: str,
                 embedder: Embedder,
                 k: int = 3,
                 rockset_collection_text_key: Optional[str] = "content",
                 ):
        self._rockset_workspace_name = rockset_workspace_name
        self._rockset_client = rockset_client
        self._query_lambda_name = query_lambda_name
        self._embedder = embedder
        self._rockset_collection_text_key = rockset_collection_text_key
        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> Prediction:
        """Search with Rockset for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            Prediction: An object containing the retrieved passages.
        """

        k = k if k is not None else self.k
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]
        passages = []
        for query in queries:
            embedding = self._embedder.embed(query)
            embedding_string = "[" + ",".join([str(num) for num in embedding]) + "]"
            api_response = self._rockset_client.QueryLambdas.execute_query_lambda_by_tag(
                query_lambda=self._query_lambda_name,
                workspace=self._rockset_workspace_name,
                tag="latest",
                parameters=[
                    QueryParameter(
                        name="search_query_embedding",
                        type="string",
                        value=embedding_string,
                    ),
                    QueryParameter(
                        name="results_limit",
                        type="int",
                        value=str(k),
                    ),
                ]
            )
            for result in api_response['results']:
                passages.append(dotdict({"long_text": result[self._rockset_collection_text_key]}))

        return passages