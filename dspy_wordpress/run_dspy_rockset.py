import os

import dspy
import weaviate
from dotenv import load_dotenv
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.util.key_loader import KeyLoader
from rockset import Regions, RocksetClient

from dspy_wordpress import WEAVIATE_CLASSNAME
from dspy_wordpress.integrations.rockset.rockset_rm import RocksetRM
from dspy_wordpress.integrations.weaviate.weaviate_v4_rm import WeaviateV4RM


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


if __name__ == '__main__':
    # Set up the OpenAI API key
    load_dotenv()

    key_loader = KeyLoader()
    rockset_api_key = key_loader.get_property("ROCKSET_API_KEY")
    openai_api_key = key_loader.get_openai_api_key()
    rockset_region = Regions.euc1a1
    workspace_name = "text_search"
    query_lambda_name = "wordpress_search_small"

    rockset = RocksetClient(host=rockset_region, api_key=rockset_api_key)

    retriever_module = RocksetRM(rockset_workspace_name=workspace_name,
                                 rockset_client=rockset,
                                 query_lambda_name=query_lambda_name,
                                 embedder=OpenAIEmbedder(api_key=openai_api_key),
                                 k=2,
                                 rockset_collection_text_key="text")

    # top_k = retriever_module("What technology is used to create our coffee assistant?").passages
    # for passage in top_k:
    #     print(passage)

    gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300, api_key=openai_api_key)
    dspy.settings.configure(lm=gpt3_turbo, rm=retriever_module)

    # qa = dspy.ChainOfThought('question,context -> answer')

    qa = RAG(num_passages=5)

    response = qa(question='What technology is used to create our coffee assistant and where can I find more '
                           'information about it?')
    # response = qa(question='Name all companies that were part of Accelerate')
    # response = qa(question='Was Bosch part of the last Accelerate?')
    # response = qa(question='What tools do I need for observability and do they run on Docker?')

    print(response)
    # print(f"Rationale: {response.rationale}")
    # print(f"Answer: {response.answer}")

    print(gpt3_turbo.history)
