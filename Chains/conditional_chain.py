from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch,
    RunnableLambda
)
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="The sentiment of the review"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    input_variables=['review'],
    template="""
Based on the following review, classify the sentiment.

Review:
{review}

{format_instructions}
""",
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive review:\n{review}",
    input_variables=['review']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative review:\n{review}",
    input_variables=['review']
)

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == 'positive',
        RunnableLambda(
            lambda x: prompt2.invoke({"review": x.review})
        ) | model | parser
    ),
    (
        lambda x: x.sentiment == 'negative',
        RunnableLambda(
            lambda x: prompt3.invoke({"review": x.review})
        ) | model | parser
    ),
    RunnableLambda(
        lambda x: "Thank you for your feedback. We appreciate your input and will use it to improve our services."
    )
)

final_chain = RunnableParallel(
    review=lambda x: x["review"],
    sentiment=classifier_chain
) | branch_chain

result = final_chain.invoke({
    "review": "The product quality is worst and the customer service was excellent!"
})

print(result)