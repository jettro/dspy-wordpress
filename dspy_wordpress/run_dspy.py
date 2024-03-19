import os
from pathlib import Path

import dspy
import weaviate
from dotenv import load_dotenv
from dspy import Retrieve
from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.rag.embedding.local.onnx_embedder import OnnxEmbedder
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rockset import Regions, RocksetClient

from dspy_wordpress import WEAVIATE_CLASSNAME
from dspy_wordpress.integrations.local.local_rm import LocalRM
from dspy_wordpress.integrations.rockset.rockset_rm import RocksetRM
from dspy_wordpress.integrations.weaviate.weaviate_v4_rm import WeaviateV4RM
from dspy_wordpress.util.wordpress_jsonl_reader import WordpressJsonlReader


class GenerateAnswer(dspy.Signature):
    """Answer questions with short answers using just a few sentences."""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Short answer of one or a few sentences.")


class RAG(dspy.Module):
    """Retrieve, Answer, Generate module."""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(question=question, context=context)
        return dspy.Prediction(answer=prediction.answer, context=context)


def retriever_module(name: str, _openai_api_key) -> Retrieve:
    if name == "weaviate":
        weaviate_api_key = os.environ.get('WEAVIATE_API_KEY')
        weaviate_url = os.environ.get('WEAVIATE_URL')

        client = weaviate.connect_to_wcs(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            headers={"X-OpenAI-Api-Key": _openai_api_key}
        )

        return WeaviateV4RM(weaviate_collection_name=WEAVIATE_CLASSNAME,
                            weaviate_client=client,
                            weaviate_collection_text_key="text",
                            k=2)
    elif name == "rockset":
        rockset_api_key = os.environ.get("ROCKSET_API_KEY")
        rockset_region = Regions.euc1a1
        workspace_name = "text_search"
        query_lambda_name = "wordpress_search_small"

        rockset = RocksetClient(host=rockset_region, api_key=rockset_api_key)

        return RocksetRM(rockset_workspace_name=workspace_name,
                         rockset_client=rockset,
                         query_lambda_name=query_lambda_name,
                         embedder=OpenAIEmbedder(api_key=openai_api_key),
                         k=2,
                         rockset_collection_text_key="text")
    elif name == "local":
        content_store = InternalContentStore(embedder=OnnxEmbedder())
        indexing_service = IndexingService(content_store=content_store)
        splitter = MaxTokenSplitter(max_tokens=200, model=DEFAULT_EMBEDDING_MODEL)
        directory = os.getcwd()
        file_path = Path(os.path.join(directory, "../data", 'two_documents.jsonl'))
        content_reader = WordpressJsonlReader(file=file_path)

        indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

        return LocalRM(content_store=content_store, k=2)
    else:
        raise ValueError(f"Unknown retriever: {name}")


if __name__ == '__main__':
    load_dotenv()

    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Setup the minimal components required by DSPy: Language Model and the Retriever.
    retriever_module = retriever_module("weaviate", openai_api_key)
    gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300, api_key=openai_api_key)
    dspy.settings.configure(lm=gpt3_turbo, rm=retriever_module)

    qa = RAG(num_passages=2)

    # qa = dspy.ChainOfThought('question, context -> answer')

    questions = [
        'What technology is used to create our coffee assistant and where can I find more information about it?',
        'Name all companies that were part of Accelerate',
        'Was Bosch part of the last Accelerate?',
        'What tools do I need for observability and do they run on Docker?'
    ]

    response = qa(question=questions[3])

    print(response)
    print(gpt3_turbo.history)
