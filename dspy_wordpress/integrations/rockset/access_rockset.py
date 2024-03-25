import json
import time

from rockset import RocksetClient, Regions, ApiException
from rockset.exceptions import NotFoundException
from rockset.model.field_mapping_query import FieldMappingQuery
from rockset.model.query_lambda_sql import QueryLambdaSql
from rockset.model.query_parameter import QueryParameter

from dspy_wordpress.integrations.rockset import logger_rockset


class AccessRockset:
    def __init__(self, api_key: str, api_server_region: Regions):
        self.api_server_region = api_server_region
        self.client = RocksetClient(host=api_server_region, api_key=api_key)
        logger_rockset.info(f"Rockset client created ...")

    def create_workspace(self, name: str):
        try:
            self.client.Workspaces.get(workspace=name)
            logger_rockset.info(f"Workspace {name} already exists.")
            return
        except NotFoundException as e:
            logger_rockset.info(f"Workspace {name} does not exist.")

        try:
            logger_rockset.info(f"Creating workspace {name} ...")
            self.client.Workspaces.create(name=name)
            logger_rockset.info(f"Workspace {name} created.")
        except Exception as e:
            logger_rockset.error(e)

    def create_collection(self, workspace: str, name: str, transformation_query: str):
        try:
            self.client.Collections.get(workspace=workspace, collection=name)
            logger_rockset.info(f"Collection {name} already exists in workspace {workspace}.")
            return
        except NotFoundException as e:
            logger_rockset.info(f"Collection {name} does not exist in workspace {workspace}.")

        try:
            logger_rockset.info(f"Creating collection {name} in workspace {workspace} ...")
            self.client.Collections.create(
                workspace=workspace,
                name=name,
                field_mapping_query=FieldMappingQuery(sql=transformation_query),
            )
            logger_rockset.info(f"Collection {name} created.")
            self.__wait_for_collection_ready(workspace=workspace, name=name)
        except Exception as e:
            logger_rockset.error(e)

    def __wait_for_collection_ready(self, workspace: str, name: str, max_attempts=20):
        logger_rockset.info(f"Waiting for the `{name}` collection to be `Ready` (~5 minutes)...")
        for attempt in range(max_attempts):
            api_response = self.client.Collections.get(collection=name, workspace=workspace)

            if api_response.data.status == 'READY':
                logger_rockset.info(f"The `{name}` collection is ready to be queried!\n")
                return
            else:
                time.sleep(30)

        logger_rockset.info(f"The `{name}` collection is still not ready. Check collection status in console.")

    def create_similarity_index(self, workspace: str, collection: str, index_name: str, embedding_field: str):
        logger_rockset.info(f"Creating `{index_name}` index for the `{collection}` collection...")

        # This is a DDL Command that will build a new index (similarity index) that we need for vector search
        # We are building the index using the FAISS IVF index with 256 centroids
        query = f"""
        CREATE
            SIMILARITY INDEX {workspace}.{index_name}
        ON
            FIELD {workspace}.{collection}:{embedding_field} DIMENSION 1536 AS 'faiss::IVF10,Flat';
        """
        try:
            res = self.client.sql(query=query)
            logger_rockset.info(f"Index `{index_name}` created!")
            self.__wait_for_index_ready(workspace=workspace, index_name=index_name)
        except ApiException as e:
            logger_rockset.error("Exception when creating similarity index: %s\n" % json.loads(e.body))

    def __wait_for_index_ready(self, workspace: str, index_name: str, max_attempts=10):
        logger_rockset.info(f"Waiting for the `{index_name}` index to be `Ready` (~1 minute)...")

        # We will be querying _system to check on the status of the index build
        query = f"""
        SELECT
            index_status
        FROM
            _system.similarity_index
        WHERE
            workspace = '{workspace}'
            and name = '{index_name}'
        """

        for attempt in range(max_attempts):
            api_response = self.client.sql(query=query)

            if api_response['results'][0]['index_status'] == 'READY':
                logger_rockset.info(f"The `{index_name}` is ready to be queried!\n")
                return
            else:
                logger_rockset.info(f"Sleep for 10 seconds: {api_response['results'][0]['index_status']}")
                time.sleep(10)
        logger_rockset.info(f"The `{index_name}` index is still not ready. Check status in console.")

    def add_document(self, workspace: str, collection: str, document: dict):
        try:
            documents = self.client.Documents.add_documents(workspace=workspace, collection=collection, data=[document])
            result = documents['data'][0]
            if result['error']:
                logger_rockset.error(f"Error when adding document: {result['error']}")
                return False
            else:
                logger_rockset.info(f"Document status: {result['status']}")
                return True
        except ApiException as e:
            logger_rockset.error("Exception when adding document: %s\n" % json.loads(e.body))

    def create_query_lambda(self, workspace: str, collection: str, query_lambda_name: str):
        description = ("Vector search (specifically Approximate Nearest Neighbors). Looking for similar texts as "
                       "search_query_embedding")
        query = f"""
        SELECT
            title,
            APPROX_DOT_PRODUCT(
                JSON_PARSE(:search_query_embedding),
                chunk_embedding
            ) as similarity,
            document_id,
            chunk_id,
            text
        FROM
            {workspace}.{collection} HINT(access_path=index_similarity_search)
        ORDER BY
            similarity DESC
        LIMIT
            :results_limit
        """

        try:
            logger_rockset.info(f"Creating query lambda `{query_lambda_name}`...")
            api_response = self.client.QueryLambdas.create_query_lambda(
                name=query_lambda_name,
                workspace=workspace,
                sql=QueryLambdaSql(
                    query=query,
                ),
            )
            logger_rockset.info(f"Query lambda `{query_lambda_name}` created!\n")
        except ApiException as e:
            logger_rockset.error(f"Exception when creating query lambda: %s\n" % json.loads(e.body))

    def query_lambda(self, workspace: str, query_lambda_name: str, embedding: list[float], results_limit: int = 3):
        try:
            logger_rockset.info(f"Executing semantic search query from search query embedding...")
            embedding_string = "[" + ",".join([str(num) for num in embedding]) + "]"
            api_response = self.client.QueryLambdas.execute_query_lambda_by_tag(
                query_lambda=query_lambda_name,
                workspace=workspace,
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
                        value=str(results_limit),
                    ),
                ]
            )
            return api_response
        except ApiException as e:
            logger_rockset.error(f"Exception when executing query lambda: %s\n" % json.loads(e.body))