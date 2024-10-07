"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.
© 2024 Massachusetts Institute of Technology.
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

""" 
This file contains two DungeonMaster classes, which simulate situations that an LLM may encounter
and evaluate its responses to them. 

The SetChoices variant assumes a fill-in-the-blank style encounter where the responses follow a
prescribed format and blanks are filled from a given list. The responses are then evaluated against
a list of acceptable and unacceptable conditions, resulting in a binary ACCEPTED or REJECTED result.

The OpenEnded variant allows the LLM responses to be freeform (though they may be somewhat)
constrained by the prompt, and are evaluated by ROUGE score.

The goal of the game is to learn the preferences inherent in the world, either by logical deduction
(SetChoices) or by matching the training set as closely as possible (OpenEnded).
"""

import random
from abc import ABC, abstractmethod
import re
import yaml

from torchmetrics.text.rouge import ROUGEScore

from tree_functions import (
    build_combinations_tree,
    remove_node,
    count_leaf_nodes,
    traverse_tree,
    max_tree_depth,
)


class DungeonMaster(ABC):
    """This class is a abstract DungeonMaster class that deals with 1) contexts, 2) offers, and 3) preferences."""

    @abstractmethod
    def print_current_context(self):
        """Return the current context as a string."""
        raise NotImplementedError

    @abstractmethod
    def print_current_offer(self):
        """Return the current offer as a string."""
        raise NotImplementedError

    @abstractmethod
    def next_context(self):
        """Record current context and advance to the next context."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_offer(self, offer: str, string_output=False, specific_context=None):
        """Evaluates "offers" of within a given context."""
        raise NotImplementedError


class DungeonMasterSetChoices(DungeonMaster):
    """This class holds the possible contexts and offers in the environment, along with the actual
    preferences. Contexts stay the same until they are "advanced" using next_context().
    """

    def __init__(
        self,
        possible_contexts: list,
        possible_offers: list,
        context_format: str,
        offer_format: str,
        preferences: dict,
        seed: int = 256,
        allow_unsatisfiable_contexts: bool = False,
    ):
        """Parameters:
        possible_contexts contains a list of lists of strings, where each inner list contains
            possible options for context_format
        possible_offers contains a list of lists of strings, where each inner list contains
            possible options for offer_format
        context_format: string with {} blanks to be filled in by items from possible_contexts
        offer_format: string with {} blanks to be filled in by items from possible_offers
        preferences: dictionary, {string:list[string]} mapping an offer item to one or more
            context items
        seed: int, random seed
        context_history: list[context], history of contexts

        Example:
            possible_contexts = [["Tom", "Cam"], ["noon", "midnight"]]
            possible_offers = [["hot", "cold"], ["soup", "sandwich"]]
            context_format = "You find {} at {}."
            offer_format = "You offer a {} {}."

            The above could resolve to "You find Tom at noon." and "You offer a cold soup."
        """

        self.possible_contexts = possible_contexts
        self.possible_offers = possible_offers
        self.context_format = context_format
        self.offer_format = offer_format
        self.preferences = preferences
        self.random_seed = seed
        random.seed(seed)
        self.current_context = [random.choice(obj) for obj in self.possible_contexts]
        self.current_offer = []
        self.context_history = []

        if not allow_unsatisfiable_contexts:
            self.validate_preferences_and_contexts()

    def print_current_context(self):
        """Return the current context as a string."""
        return self.context_format.format(*self.current_context)

    def print_current_offer(self):
        """Return the current offer as a string."""
        return self.offer_format.format(*self.current_offer)

    def next_context(self):
        """Record current context and advance to the next context."""
        self.context_history.append(self.current_context)
        self.current_context = [random.choice(obj) for obj in self.possible_contexts]

    def evaluate_offer(self, offer: str, string_output=False, specific_context=None):
        """Parameters:
            offer: string of offer
            string_output: bool - whether the output is a string or not
                (default = False -> boolean)
            specific_context: [who, where, when] of the current context
                (default = None -> self.current_context)
        Output:
            result: bool for whether an offer was accepted in the current context

        This function evaluates "offers" of some food at some temperature.
        The robot provides as input an OFFER of [temperature, food] to a CONTEXT of [who, where,
            when] and gets a RESPONSE of ACCEPT/REJECT.
        """
        if specific_context is None:
            specific_context = self.current_context

        # extract properties after splitting out the reason, if any is offered
        self.current_offer = extract_properties(
            offer.split("because")[0], *self.possible_offers
        )

        # offer statement does not contain proper items from offer categories - do not
        # evaluate the offer
        all_valid = True
        for item in self.current_offer:
            if item is None:
                all_valid = False
        if not all_valid:
            if string_output:
                return "INVALID OFFER"
            return False

        # offer statement contains proper items from offer categories
        offer_accepted = False
        for preference in self.preferences:
            if preference in offer:
                contexts_in_which_preference_applies = self.preferences[preference]
                for subcontext in specific_context:
                    if subcontext in contexts_in_which_preference_applies:
                        offer_accepted = True
                        break

        if string_output:
            return "ACCEPTED" if offer_accepted else "REJECTED"
        return offer_accepted

    def validate_preferences_and_contexts(self):
        """
        Check that the preference list contains enough preferences so that all contexts can be
        satisfied.
        """

        tree = build_combinations_tree(self.possible_contexts)
        initial_tree_depth = max_tree_depth(tree)
        tree_depth = initial_tree_depth
        for _, contexts in self.preferences.items():
            for context in contexts:
                tree = remove_node(tree, context)
                node_count = count_leaf_nodes(tree)
                if node_count <= 1:
                    break
                tree_depth = max_tree_depth(tree)
                if tree_depth < initial_tree_depth:
                    # if the tree is still there, but the max depth has decreased, we've eliminated
                    # all possibilities
                    break
            if node_count <= 1:
                break
        assert (
            node_count == 1 or tree_depth < initial_tree_depth
        ), f"Warning: there are unsatisfiable contexts (above) {traverse_tree(tree)}"
        print("All contexts can be satisfied by the given preferences.")


class DungeonMasterOpenEnded(DungeonMaster):
    """This class is a DungeonMaster subclass that deals with 1) contexts, 2) offers, and 3)
    preferences, all of which are open-ended text responses."""

    def __init__(self, contexts, ground_truths, preferences, evaluation_method):
        """Store a function to be used in evaluating responses."""
        self.evaluation_method = evaluation_method
        self.contexts = contexts
        self.ground_truths = ground_truths
        self.context_counter = 0
        self.preferences = preferences
        self.current_context = self.contexts[self.context_counter]["input"]
        self.current_ground_truth = self.ground_truths["golds"][self.context_counter][
            "output"
        ]
        self.current_preference = ""
        self.current_offer = ""
        self.context_history = []

    def print_current_context(self):
        """Return the current context as a string."""
        return self.current_context

    def print_current_offer(self):
        """Return the current offer as a string."""
        return self.current_offer

    def print_current_ground_truth(self):
        """Return the current ground truth as a string."""
        return self.current_ground_truth

    def next_context(self):
        """Increase counter for context"""
        self.context_counter += 1
        self.current_context = self.contexts[self.context_counter]["input"]
        self.current_ground_truth = self.ground_truths["golds"][self.context_counter][
            "output"
        ]  # these identifiers are unique to the LaMP dataset, can be adjusted to your specific dataset
        if self.preferences is not None:
            self.current_preference = self.preferences[self.context_counter]

    def evaluate_offer(
        self, offer: str, ground_truth: str, *, answer_id_string: str = None
    ):
        """Evaluates "offers" of within a given context."""

        updated_offer = []
        for o_ in offer:
            # extracting headline if prefix is specified
            if answer_id_string is None:
                extracted_str = o_
            else:
                extracted_str = re.search(answer_id_string + r": (.*?)($|\n)", o_)
                if extracted_str is None:
                    extracted_str = o_
                else:
                    extracted_str = extracted_str.group(1)

            # removing quotation marks
            if len(extracted_str) == 0:
                pass
            elif extracted_str[0] == '"':
                extracted_str = extracted_str[1:-1]

            updated_offer.append(extracted_str)

        rouge = ROUGEScore()
        return rouge(updated_offer, ground_truth)


def simulate_offer(
    dm: DungeonMaster, iterations: int = 50, print_results: bool = False
) -> tuple:
    """Parameters:
        dm: DungeonMaster instance
        iterations: int, number of times to make an offer to in a context
        print_results: boolean, indicating whether to print the action and the result
    Output:
        contexts: list, shape (iterations x 3) list of lists of [*context]
        offers: list, shape (iterations x 2) list of lists of [*offer]
        results: list, shape (iterations) list of bools for whether an offer was accepted in the
            corresponding context

    This function simulates a series of "offers".
    The robot provides as input an OFFER to a CONTEXT and gets a RESPONSE of ACCEPT/REJECT.
    """
    contexts = []
    offers = []
    results = []
    for _ in range(iterations):

        contexts.append(dm.current_context)

        offer = [random.choice(obj) for obj in dm.possible_offers]
        offers.append(offer)

        offer_accepted = dm.evaluate_offer(offer)

        result = "ACCEPTED" if offer_accepted else "REJECTED"
        results.append(result)
        if print_results:
            sentence = dm.print_current_context() + " " + dm.print_current_offer()
            print(f"{sentence} {result}")

        dm.next_context()

    return contexts, offers, results


def extract_properties(offer_statement: str, *argv) -> list:
    """Given a natural language statement and lists of possible properties
    Return the properties contained in the statement
    """

    properties = [None] * len(argv)
    possible_properties = argv

    for i, prop in enumerate(possible_properties):
        for obj in prop:
            if obj in offer_statement:
                properties[i] = obj
                break

    return properties


def check_and_load_yaml(context_yaml_path: str, preferences_yaml_path: str) -> tuple:
    """
    Reads data from YAML files and checks if preferences match the available categories.

    Args:
        context_yaml_path (str): Path to the YAML file containing context categories.
        preferences_yaml_path (str): Path to the YAML file containing user preferences.

    Returns:
        tuple: A tuple containing three dictionaries:
            - context_categories: Dictionary containing context categories and their
            corresponding values.
            - offer_categories: Dictionary containing offer categories and their corresponding
            values.
            - preferences: Dictionary containing user preferences.

    Raises:
        AssertionError: If a preference key or value is not found in the corresponding categories.
    """

    # Reading data from YAML file
    with open(context_yaml_path, "r", encoding="utf-8") as file:
        setting_and_cast = yaml.safe_load(file)
        context_categories = setting_and_cast["context_categories"]
        offer_categories = setting_and_cast["offer_categories"]

    with open(preferences_yaml_path, "r", encoding="utf-8") as file:
        preferences = yaml.safe_load(file)

    # checking that preferences and setting_and_cast yaml files match
    for key, value in preferences.items():

        # check that the key is in one of the offer categories
        key_in_preferences = False
        for category in offer_categories.keys():
            if key in offer_categories[category]:
                key_in_preferences = True
                break
        assert key_in_preferences, f"{key} not in offer categories"

        # check that the sub-value is in one of the context categories
        for v in value:
            value_in_preferences = False
            for category in context_categories.keys():
                if v in context_categories[category]:
                    value_in_preferences = True
                    break
            assert value_in_preferences, f"{v} not in context categories"

    return context_categories, offer_categories, preferences
