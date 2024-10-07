"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

""" Abstract Bot superclasses and instantiated subclasses representing different agent types we 
are testing."""

from abc import ABC, abstractmethod
import random
from langchain_openai import ChatOpenAI

from langchain_setup import (
    setup_offer_chain,
    setup_memory_summary_chain,
    build_retriever,
    setup_RAG_offer_chain,
)


class Bot(ABC):
    """Abstract bot class enforcing required methods for each bot variant."""

    @abstractmethod
    def __init__(self, bot_name):
        self.name = bot_name


class SetChoicesBot(Bot):
    """Abstract bot class enforcing required methods for each bot variant that uses a fixed
    set of choices in its contexts and responses."""

    @abstractmethod
    def make_offer(self, offer_categories: dict, prompt_str: str = None) -> str:
        """Make an offer based on bot's own methods."""

    @abstractmethod
    def update_memory(self, new_memory, replace=False) -> None:
        """Update bot memory for use with context (if applicable)."""


class OpenEndedBot(Bot):
    """Abstract bot class enforcing required methods for each bot variant that allows for
    open-ended contexts and responses."""

    @abstractmethod
    def make_offer(self, prompt_str: str = None) -> str:
        """Make an offer based on bot's own methods."""

    @abstractmethod
    def update_internal_memory(self, new_memory, replace=False) -> None:
        """
        Update bot memory for use with conversation history.

        Args:
            new_memory: Latest interaction with the LLM that is recorded.
            replace: Whether the external memory is added to or written over.

        Returns:
            None.
        """

    @abstractmethod
    def update_external_memory(
        self,
        new_memory: list,
        new_memory_full_list: list = None,
        new_memory_index: int = None,
        replace: bool = False,
    ) -> None:
        """
        Update bot memory for use with context (if applicable).

        Args:
            new_memory: External documents to be retained in the bot's memory.
            new_memory_full_list: In the case that a full dataset is used and there's are external documents for each entity in the list, the whole list can also be provided (eg. in the case of random bot).
            new_memory_index: The index in the new_memory_full_list that has the same documents as new_memory.
            replace: Whether the external memory is added to or written over.

        Returns:
            None.
        """


def fill_template(str_template: str, *argv: str) -> str:
    """
    Fill in the template string with the provided arguments.

    Args:
        str_template: The template string containing curly braces to be replaced.
        *argv: The arguments to replace the curly braces in the template string.

    Returns:
        The filled template string with arguments replaced.

    Raises:
        ValueError: If the number of arguments does not match the number of curly braces in the template.
    """
    # Count the number of curly braces in the template string
    num_braces = str_template.count("{}")

    # Check if the number of arguments matches the number of curly braces
    if len(argv) != num_braces:
        raise ValueError(
            "Number of arguments does not match the number of curly braces in the template."
        )

    # Replace each curly brace pair with its corresponding argument
    filled_template = str_template
    for arg in argv:
        filled_template = filled_template.replace("{}", str(arg), 1)

    return filled_template


class RandomBot(SetChoicesBot):
    """
    Bot that randomly chooses an offer from a set of choices. Since the offer format is
    hard-coded, it will always produce offers that match string specification. RandomBot
    does not instantiate update_memory().
    """

    def __init__(self, response_template: str):
        """
        The RandomBot init only uses the response template to generate a random response.
        """
        super().__init__("RandomBot")
        self.response_template = response_template

    def update_memory(self, new_memory, replace=False) -> None:
        """The RandomBot update_memory doesn't actually need to do anything, it's just here for
        consistency."""
        return super().update_memory(new_memory)

    def make_offer(self, offer_categories: dict, prompt_str: str = None) -> str:
        """Make an offer randomly (ignores the prompt_str)."""
        offer_selections = [
            random.choice(offer_categories[key])
            for key in list(offer_categories.keys())
        ]
        if "because" in self.response_template:
            offer_selections.append("I chose randomly")

        # construct the response string
        response_str = fill_template(self.response_template, *offer_selections)

        return response_str


class LLMWithEntireHistoryBot(SetChoicesBot):
    """Bot that uses an LLM with the entire interaction history as context.
    Since the entire offer string is LLM-generated, it may produce offers that do not
    match string specification.
    """

    def __init__(self, llm: ChatOpenAI, initial_prompt: str, offer_template: str):
        super().__init__("LLMWithEntireHistoryBot")
        self.offer_chain = setup_offer_chain(llm, offer_template)
        self.initial_prompt = initial_prompt
        self.chat_history = ""

    def update_memory(self, new_memory, replace=False) -> None:
        if not replace:
            self.chat_history += f"\n{new_memory}"
        else:
            self.chat_history = new_memory

    def make_offer(self, offer_categories: dict, prompt_str: str = None) -> str:
        """Make an offer conditioned on the entire interaction history."""
        # a context was given - agent should make an offer
        langchain_template_options = offer_categories.copy()
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.chat_history}"
        )
        langchain_template_options["prompt"] = prompt_str
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        return llm_final_response.content


class LLMWithPeriodicSummaryBot(SetChoicesBot):
    """Bot that uses an LLM with only a periodically-summarized history of the
    entire chat as context.
    Since the entire offer string is LLM-generated, it may produce offers that do not
    match string specification.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        initial_prompt: str,
        offer_template: str,
        summarization_period: int,
    ):
        super().__init__("LLMWithPeriodicSummaryBot")
        self.offer_chain = setup_offer_chain(llm, offer_template)
        self.summary_chain = setup_memory_summary_chain(llm)
        self.initial_prompt = initial_prompt
        self.chat_history = ""
        self.summary = "No summary available."
        self.summarization_period = summarization_period
        self.contexts_encountered = 0

    def update_memory(self, new_memory, replace=False) -> None:
        if not replace:
            self.chat_history += f"\n{new_memory}"
        else:
            self.chat_history = new_memory

    def make_offer(self, offer_categories: dict, prompt_str: str = None) -> str:
        """Make an offer conditioned on the most recent summary."""

        # a context was given - agent should make an offer
        langchain_template_options = offer_categories.copy()
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.summary}",
        )
        langchain_template_options["prompt"] = prompt_str
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        self.contexts_encountered += 1
        if self.contexts_encountered % self.summarization_period == 0:
            self.summary = self.summary_chain.invoke(
                {"context": self.chat_history}
            ).content

        return llm_final_response.content


class LLMWithNoHistoryOpenEndedBot(OpenEndedBot):
    """OpenEndedBot that uses an LLM with the no external context / documents."""

    def __init__(
        self,
        llm: ChatOpenAI,
        initial_prompt: str,
        offer_template: str,
        name: str = "LLMWithNoHistoryOpenEndedBot",
    ):
        """
        Args:
            llm: Model used (eg. GPT-3.5).
            initial_prompt: System prompt.
            offer_template: Prompt template to send to the model.
            name: Name of the bot.
        """

        super().__init__(name)
        self.offer_chain = setup_offer_chain(llm, offer_template)
        self.initial_prompt = initial_prompt
        self.chat_history = ""
        self.ref_info = []

    def update_internal_memory(self, new_memory: str, replace=True) -> None:
        if replace:
            self.chat_history = new_memory
        else:
            self.chat_history += f"\n{new_memory}"

    def update_external_memory(
        self,
        new_memory: list,
        new_memory_full_list: list = None,
        new_memory_index: int = None,
        replace=True,
    ) -> None:
        super().update_external_memory(
            new_memory, new_memory_full_list, new_memory_index, replace
        )

    def make_offer(self, prompt_str: str = None) -> str:
        """
        Make an offer conditioned on no external history provided. 'prompt_str' is the specific prompt you want to send to the model.
        """
        langchain_template_options = {}
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.chat_history}"
        )
        langchain_template_options["prompt"] = prompt_str
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        return llm_final_response.content


class LLMWithEntireHistoryOpenEndedBot(OpenEndedBot):
    """OpenEndedBot that uses an LLM with the entire interaction history as context."""

    def __init__(
        self,
        llm: ChatOpenAI,
        initial_prompt: str,
        offer_template: str,
        name: str = "LLMWithEntireHistoryOpenEndedBot",
    ):
        """
        Args:
            llm: Model used (eg. GPT-3.5).
            initial_prompt: System prompt.
            offer_template: Prompt template to send to the model.
            name: Name of the bot.
        """

        super().__init__(name)
        self.offer_chain = setup_offer_chain(llm, offer_template)
        self.initial_prompt = initial_prompt
        self.chat_history = ""
        self.ref_info = []

    def update_internal_memory(self, new_memory: str, replace=True) -> None:
        if replace:
            self.chat_history = new_memory
        else:
            self.chat_history += f"\n{new_memory}"

    def update_external_memory(
        self,
        new_memory: list,
        new_memory_full_list: list = None,
        new_memory_index: int = None,
        replace=True,
    ) -> None:
        """
        'new_memory_full_list' and 'new_memory_index' are not used.
        """

        if replace:
            self.ref_info = new_memory
        else:
            self.ref_info += new_memory

    def make_offer(self, prompt_str: str, k: int = 50) -> str:
        """
        Makes an offer conditioned on the "entire" external history (up to some limit k), essentially whatever fits in token length.
        """

        langchain_template_options = {}
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.chat_history}"
        )
        profile_memory = "\n\nHere is the history of accepted texts and their coorresponding titles:\n"
        for profile_entry in self.ref_info[:k]:
            profile_memory += f"article text: {profile_entry['text']}\n"
            profile_memory += f"article title: {profile_entry['title']}\n\n"  # these identifiers are unique to the LaMP dataset, can be adjusted to your specific dataset
        langchain_template_options["prompt"] = prompt_str + profile_memory
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        return llm_final_response.content


class LLMWithRetrievedHistoryOpenEndedBot(OpenEndedBot):
    """OpenEndedBot that uses an LLM with the retrieved interaction history as context."""

    def __init__(
        self,
        llm: ChatOpenAI,
        initial_prompt: str,
        offer_template: str,
        name: str = "LLMWithRetrievedHistoryOpenEndedBot",
    ):
        """
        Args:
            llm: Model used (eg. GPT-3.5).
            initial_prompt: System prompt.
            offer_template: Prompt template to send to the model.
            name: Name of the bot.
        """

        super().__init__(name)
        self.offer_chain = None
        self.offer_template = offer_template
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.chat_history = ""
        self.ref_info = []

    def update_internal_memory(self, new_memory: str, replace=True) -> None:
        if replace:
            self.chat_history = new_memory
        else:
            self.chat_history += f"\n{new_memory}"

    def update_external_memory(
        self,
        new_memory: list,
        new_memory_full_list: list = None,
        new_memory_index: int = None,
        replace=True,
    ) -> None:
        """
        'new_memory_full_list' and 'new_memory_index' are not used.
        """

        if replace:
            self.ref_info = new_memory
        else:
            self.ref_info += new_memory

        self.retriever = build_retriever(self.ref_info)

    def make_offer(self, prompt_str: str = None) -> str:
        """
        Make an offer conditioned on the retrieved interaction history (up to some limit k).
        """

        langchain_template_options = {}
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.chat_history}"
        )

        langchain_template_options["prompt"] = prompt_str
        self.offer_chain = setup_RAG_offer_chain(
            self.llm, self.offer_template, self.retriever
        )
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        self.retriever.vectorstore.delete_collection()

        return llm_final_response


class LLMWithRandomHistoryOpenEndedBot(OpenEndedBot):
    """OpenEndedBot that uses an LLM with a randomly-retrieved piece of interaction history as context."""

    def __init__(
        self,
        llm: ChatOpenAI,
        initial_prompt: str,
        offer_template: str,
        name: str = "LLMWithRandomHistoryOpenEndedBot",
    ):
        """
        Args:
            llm: Model used (eg. GPT-3.5).
            initial_prompt: System prompt.
            offer_template: Prompt template to send to the model.
            name: Name of the bot.
        """

        super().__init__(name)
        self.offer_chain = setup_offer_chain(llm, offer_template)
        self.offer_template = offer_template
        self.llm = llm
        self.initial_prompt = initial_prompt
        self.chat_history = ""
        self.ref_info = []

    def update_internal_memory(self, new_memory: str, replace=True) -> None:
        if replace:
            self.chat_history = new_memory
        else:
            self.chat_history += f"\n{new_memory}"

    def update_external_memory(
        self,
        new_memory: list,
        new_memory_full_list: list = None,
        new_memory_index: int = None,
        replace=True,
    ) -> None:
        super().update_external_memory(
            new_memory, new_memory_full_list, new_memory_index, replace
        )

        if self.ref_info == []:
            self.ref_info = new_memory_full_list

    def make_offer(self, prompt_str: str, k: int = 5) -> str:
        """
        Make an offer conditioned on randomly selected k=5 interaction history (up to some limit k).
        """
        langchain_template_options = {}
        langchain_template_options["context"] = (
            f"{self.initial_prompt}\n{self.chat_history}"
        )

        profile_memory = ""
        for _ in range(k):
            prompt_ind = random.choice(range(len(self.ref_info)))
            profile = self.ref_info[prompt_ind]["profile"]
            doc_ind = random.choice(range(len(profile)))
            text, title = (
                profile[doc_ind]["text"],
                profile[doc_ind]["title"],
            )
            profile_memory += f"article text: {text}\n"
            profile_memory += f"article title: {title}\n\n"

        langchain_template_options["retrieved_data"] = profile_memory

        langchain_template_options["prompt"] = prompt_str
        llm_final_response = self.offer_chain.invoke(langchain_template_options)

        return llm_final_response.content
