import logging
import os
import sys
from pathlib import Path

from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate import chunk_collection
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_content_store import WeaviateContentStore
from rag4p.util.key_loader import KeyLoader

from dspy_wordpress import WEAVIATE_CLASSNAME
from dspy_wordpress.integrations.weaviate.wordpress_collection import wordpress_collection_properties
from dspy_wordpress.util.wordpress_jsonl_reader import WordpressJsonlReader

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    key_loader = KeyLoader()
    access_weaviate = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    access_weaviate.force_create_collection(collection_name=WEAVIATE_CLASSNAME,
                                            properties=chunk_collection.weaviate_properties(
                                                additional_properties=wordpress_collection_properties()
                                            ))

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    content_store = WeaviateContentStore(weaviate_access=access_weaviate, embedder=embedder,
                                         collection_name=WEAVIATE_CLASSNAME)
    splitter = MaxTokenSplitter(max_tokens=200, model=DEFAULT_EMBEDDING_MODEL)
    indexing_service = IndexingService(content_store=content_store)

    directory = os.getcwd()
    file_path = Path(os.path.join(directory, "../data", 'two_documents.jsonl'))
    content_reader = WordpressJsonlReader(file=file_path)

    indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

    access_weaviate.close()
