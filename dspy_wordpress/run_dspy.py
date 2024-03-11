import os

import dspy
import weaviate
from dotenv import load_dotenv

from dspy_wordpress import WEAVIATE_CLASSNAME
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

    openai_api_key = os.environ.get('OPENAI_API_KEY')
    weaviate_api_key = os.environ.get('WEAVIATE_API_KEY')
    weaviate_url = os.environ.get('WEAVIATE_URL')

    client = weaviate.connect_to_wcs(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        headers={"X-OpenAI-Api-Key": openai_api_key}
    )

    retriever_module = WeaviateV4RM(weaviate_collection_name=WEAVIATE_CLASSNAME,
                                    weaviate_client=client,
                                    weaviate_collection_text_key="text",
                                    k=2)

    # top_k = retriever_module("What technology is used to create our coffee assistant?").passages
    # for passage in top_k:
    #     print(passage)

    gpt3_turbo = dspy.OpenAI(model='gpt-3.5-turbo-1106', max_tokens=300, api_key=openai_api_key)
    dspy.settings.configure(lm=gpt3_turbo, rm=retriever_module)

    # qa = dspy.ChainOfThought('question,context -> answer')

    qa = RAG(num_passages=2)

    # response = qa(question='What technology is used to create our coffee assistant and where can I find more '
    #                        'information about it?')
    # response = qa(question='Name all companies that were part of Accelerate')
    # response = qa(question='Was Bosch part of the last Accelerate?')
    response = qa(question='What tools do I need for observability and do they run on Docker?')

    print(response)
    # print(f"Rationale: {response.rationale}")
    # print(f"Answer: {response.answer}")

    print(gpt3_turbo.history)
