"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

""" Functions to set up LangChain chains.
"""

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def setup_offer_chain(llm, template):
    """Set up the langchain chain to deal with offers."""

    answer_prompt = ChatPromptTemplate.from_template(template)

    return answer_prompt | llm


def setup_memory_summary_chain(llm):
    """Set up the langchain chain to summarize the working memory."""

    template = """Summarize the following set of encounters, creating a brief synopsis of what you have learned about the game.
    Be logical about what information you are summarizing, and be as specific as you can about the correlations that you find.
    
    Here is the set of encounters:
    {context}
    """

    summary_prompt = ChatPromptTemplate.from_template(template)

    return summary_prompt | llm


def setup_RAG_offer_chain(llm, template, retriever):
    """Set up the langchain chain to deal with offers."""

    answer_prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": itemgetter("context"),
            "prompt": itemgetter("prompt"),
            "retrieved_data": itemgetter("prompt") | retriever,
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_retriever(user_profile: list, k: int = 5):
    """Build the vectorstore retriever to use the top k user-specific entries.

    user_profile: a list of profile entries, each of which is a dictionary with 'text' and 'title' keys
    k: the top k values ot return
    """
    user_profile_string = [
        f"\narticle text: {profile_entry['text']}\n"
        + f"article title: {profile_entry['title']}\n\n"
        for profile_entry in user_profile
    ]

    vectorstore = Chroma.from_texts(
        texts=user_profile_string,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )
