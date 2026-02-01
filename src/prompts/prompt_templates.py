from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.utils.config import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE


def get_qa_prompt() -> PromptTemplate:
    """
    Get the main QA prompt template.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE
    )


def get_qa_with_system_prompt() -> PromptTemplate:
    """
    Get QA prompt with system instructions.
    """
    template = f"""{SYSTEM_PROMPT}

{{context}}

Question: {{question}}

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def get_conversational_prompt() -> PromptTemplate:
    """
    Get prompt for conversational QA (with chat history).
    """
    template = f"""{SYSTEM_PROMPT}

Previous conversation:
{{chat_history}}

Current context from Gale Encyclopedia:
{{context}}

Current question: {{question}}

Provide a helpful answer based on the context and conversation history.

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )


def get_summarization_prompt() -> PromptTemplate:
    """
    Get prompt for summarizing medical information.
    """
    template = """Summarize the following medical information from the Gale Encyclopedia in clear, concise language:

{text}

Summary:"""

    return PromptTemplate(
        input_variables=["text"],
        template=template
    )


def get_definition_prompt() -> PromptTemplate:
    """
    Get prompt for medical term definitions.
    """
    template = """Based on the Gale Encyclopedia of Medicine, provide a clear definition of the following medical term:

Term: {term}

Context from encyclopedia:
{context}

Definition:"""

    return PromptTemplate(
        input_variables=["term", "context"],
        template=template
    )


def format_sources(sources: list) -> str:
    """
    Format source documents for display.
    """
    if not sources:
        return ""

    formatted = "\n\n** Sources from Gale Encyclopedia:**\n"

    unique_sources = {}
    for doc in sources:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "N/A")

        if source not in unique_sources:
            unique_sources[source] = set()
        unique_sources[source].add(page)

    for i, (source, pages) in enumerate(unique_sources.items(), 1):
        pages_sorted = sorted(
            pages,
            key=lambda x: int(x) if isinstance(x, int) else 999
        )
        pages_str = ", ".join(str(p) for p in pages_sorted)
        formatted += f"\n{i}. {source} (Pages: {pages_str})"

    return formatted


def create_custom_prompt(
    system_msg: str,
    template_str: str,
    input_vars: list
) -> PromptTemplate:
    """
    Create a custom prompt template.
    """
    full_template = f"""{system_msg}

{template_str}"""

    return PromptTemplate(
        input_variables=input_vars,
        template=full_template
    )


# Example prompts
SYMPTOM_CHECKER_PROMPT = """You are analyzing symptoms from the Gale Encyclopedia of Medicine.

Context: {context}
Symptoms mentioned: {symptoms}

Provide information about possible conditions related to these symptoms, based ONLY on the encyclopedia context.
Remind the user to consult a healthcare professional.

Analysis:"""


MEDICATION_INFO_PROMPT = """You are providing medication information from the Gale Encyclopedia of Medicine.

Context: {context}
Medication: {medication}

Provide clear information about this medication including:
- What it treats
- How it works
- Common side effects (if mentioned)

Based ONLY on the encyclopedia context.

Information:"""


CONDITION_OVERVIEW_PROMPT = """You are explaining a medical condition using the Gale Encyclopedia of Medicine.

Context: {context}
Condition: {condition}

Provide a comprehensive overview including:
- Definition
- Causes
- Symptoms
- Treatment options

Based ONLY on the encyclopedia context.

Overview:"""


def main():
    print("\nTESTING PROMPT TEMPLATES\n")

    qa_prompt = get_qa_prompt()
    print("QA Prompt Template:")
    print(
        qa_prompt.format(
            context="Diabetes is a chronic disease...",
            question="What is diabetes?"
        )
    )
    print()

    test_docs = [
        Document(page_content="...", metadata={"source_file": "gale_vol1.pdf", "page_number": 45}),
        Document(page_content="...", metadata={"source_file": "gale_vol1.pdf", "page_number": 46}),
        Document(page_content="...", metadata={"source_file": "gale_vol1.pdf", "page_number": 52}),
    ]

    print("Source Formatting:")
    print(format_sources(test_docs))

    print("\n Prompt templates test successful!\n")


if __name__ == "__main__":
    main()