from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["paper", "style","length"],
    template="""
                You are an expert research assistant.

                Your task is to summarize the research paper below.

                Research Paper:
                {paper}

                Instructions:
                - Use the following style: {style}
                - Keep the response length: {length}
                - Explain complex concepts simply
                - Highlight the most important contributions
                - Maintain clarity and structure

                Generate the response now.
            """,
    validate_template=True
)
template.save('template.json')