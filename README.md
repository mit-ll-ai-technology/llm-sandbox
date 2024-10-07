# Preference Dungeon

## Overview

Preference Dungeon is a testing tool for large language models (LLMs), handling single-turn prompt and response situations.

Interaction with LLMs is mediated through the use of `Bots` (`bot_interfaces.py`), of which there are two abstract subclasses `SetChoicesBot` and `OpenEndedBot`. 

`SetChoicesBot` is used for logical reasoning tasks, where the prompt requires a response in a particular format, which involves filling in parts of the response with words from a given set, and the entire response can be judged to be acceptable or not.

`OpenEndedBot` is used for open-ended responses, where the prompt does not have to contain specific keywords, and the entire response is judged by ROGUE score.

Bots created with either form of abstract subclass can be made to respond directly to the prompt, use a variety of retrieval-augmented generation, summarization, or other prepended text approaches. Text for these are defined in the yaml files associated wit each setting.

`preference_dungeon.py` includes classes to instantiate two different `DungeonMaster` subclasses that work with the SetChoices and OpenEnded bot variants, present them to prompts, and rate their responses. SetChoices bots receive a binary ACCEPTED/REJECTED result for each response, depending on whether the logical conditions are met. OpenEnded bot responses receive a ROUGE score. The evaluation loop allows the same prompt to be presented to a collection of bots, and their responses to be evaluated, before moving on to additional prompts.

## Prompt Templates

The `prompt_templates` folder contains the yaml files that define each bot instance's expected responses, along with the logical conditions ("preferences") in the case of SetChoices bots and training data in the case of OpenEnded bots. Two scenarios, `dungeons_and_dragons` and `food_offer` are provided for SetChoices bots, and one scenario, `lamp4` (based on the LaMP-4 benchmark) is provided for OpenEnded bots. 

## Example Notebooks

Example notebooks are provided in `SetChoices_Experimentation.ipynb` and `OpenEnded_Experimentation_LaMP.ipynb`.

## Reproducing LaMP4 Responses

`prompt_templates/lamp4` contains yaml files that can be used to produce LaMP-4 headlines under different prompt formats. LaMP-4 json files should be downloaded from the LaMP repository and placed in that directory.

## Creating a New Scenario

The notebooks provide a guide for how to set up a new scenario. First decide whether your evaluation is fill-in-the-blank (SetChoices) or freeform (OpenEnded). You will need to provide prompt yaml files, bot_interface instantiations, and truth data, and optionally, background information (e.g. `data/dungeons_and_dragons/background_information.txt`).

## Distribution Statement

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
