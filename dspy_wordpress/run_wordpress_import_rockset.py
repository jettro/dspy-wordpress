import logging
import os
import sys
from pathlib import Path

from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.util.key_loader import KeyLoader
from rockset import Regions

from dspy_wordpress.integrations.rockset import logger_rockset
from dspy_wordpress.integrations.rockset.access_rockset import AccessRockset
from dspy_wordpress.integrations.rockset.rockset_content_store import RocksetContentStore
from dspy_wordpress.integrations.rockset.wordpress_collection import ingest_transformation_query
from dspy_wordpress.util.wordpress_jsonl_reader import WordpressJsonlReader


def initialise_rockset():
    rockset.create_workspace(name=workspace_name)
    rockset.create_collection(workspace=workspace_name,
                              name=collection_name,
                              transformation_query=ingest_transformation_query)
    rockset.create_similarity_index(workspace=workspace_name,
                                    collection=collection_name,
                                    index_name=similarity_index_name,
                                    embedding_field=embedding_field)

    # Insert the documents
    content_store = RocksetContentStore(rockset_access=rockset,
                                        collection_name=collection_name,
                                        workspace_name=workspace_name,
                                        embedder=OpenAIEmbedder(api_key=openai_api_key))
    indexing_service = IndexingService(content_store=content_store)
    splitter = MaxTokenSplitter(max_tokens=200, model=DEFAULT_EMBEDDING_MODEL)
    directory = os.getcwd()
    file_path = Path(os.path.join(directory, "../data", 'all_documents.jsonl'))
    content_reader = WordpressJsonlReader(file=file_path)
    indexing_service.index_documents(content_reader=content_reader, splitter=splitter)


if __name__ == '__main__':
    """
    https://docs.rockset.com/documentation/recipes/semantic-search-with-openai
    """
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger_rockset.setLevel(logging.DEBUG)

    logging.info("Starting the WordPress import to Rockset...")

    # Load parameters from environment
    key_loader = KeyLoader()
    rockset_api_key = key_loader.get_property("ROCKSET_API_KEY")
    openai_api_key = key_loader.get_openai_api_key()
    rocket_region = Regions.euc1a1

    # Specify static parameters
    workspace_name = "text_search"
    collection_name = "WordPress"
    query_lambda_name = "wordpress_search_small"
    similarity_index_name = "wordpress_embeddings_similarity_index_small"
    embedding_field = "chunk_embedding"

    # Initialise the collection
    rockset = AccessRockset(api_key=rockset_api_key, api_server_region=rocket_region)
    # rockset.create_similarity_index(workspace=workspace_name,
    #                                 collection=collection_name,
    #                                 index_name=similarity_index_name,
    #                                 embedding_field=embedding_field)

    # initialise_rockset()
    rockset.create_query_lambda(workspace=workspace_name,
                                collection=collection_name,
                                query_lambda_name=query_lambda_name)

    # search_query = "What technology is used to create our coffee assistant?"
    search_query = "What technology is used to implement observability"
    embedder = OpenAIEmbedder(api_key=openai_api_key)
    embedding = embedder.embed(search_query)

    results = rockset.query_lambda(workspace=workspace_name, query_lambda_name=query_lambda_name, embedding=embedding)

    print(results)
