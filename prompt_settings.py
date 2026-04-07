from typing import Dict
from translate import Translator
from langdetect import detect, DetectorFactory

from cat import hook, AgenticWorkflowOutput, UserMessage, RecallSettings

# to guarantee consistent results (recommended)
DetectorFactory.seed = 0

# Default prompt settings
only_local = False
disable_memory = False
custom_prefix = ""
custom_suffix = ""
custom_reply_no_memory = ""
number_of_memory_items = 5
number_of_history_items = 5
threshold = 0.5
tags = {}
metadata_or_filter = False


def get_message_language(message):
    try:
        lang = detect(message)
        return lang  # es. 'it', 'en', 'fr'
    except:
        # Fallback
        return "en"


def update_variables(settings: Dict, prompt_settings: Dict | None):
    global only_local, disable_memory, custom_prefix, custom_suffix, custom_reply_no_memory, number_of_memory_items, number_of_history_items, threshold, tags, metadata_or_filter

    prompt_settings = prompt_settings or {}

    only_local = prompt_settings.get("only_local_responses", settings["only_local_responses"])
    disable_memory = prompt_settings.get("disable_memories", settings["disable_memories"])
    custom_prefix = prompt_settings.get("prompt_prefix", settings["prompt_prefix"])
    custom_suffix = prompt_settings.get("prompt_suffix", settings["prompt_suffix"])
    custom_reply_no_memory = prompt_settings.get("reply_no_memory", settings["reply_no_memory"])
    number_of_memory_items = prompt_settings.get("number_of_memory_items", settings["number_of_memory_items"])
    number_of_history_items = prompt_settings.get("number_of_history_items", settings["number_of_history_items"])
    threshold = prompt_settings.get("threshold", settings["threshold"])
    metadata_or_filter = settings["enable_OR_condition_for_metadata_filter"]


@hook(priority=10)
async def before_cat_reads_message(user_message: UserMessage, cat):
    global tags
    settings = await cat.mad_hatter.get_plugin().load_settings()
    prompt_settings = None
    tags = {}
    if "prompt_settings" in user_message:
        prompt_settings = user_message.get("prompt_settings")
    if "tags" in user_message:
        tags = user_message.get("tags")
    update_variables(settings, prompt_settings)
    return user_message


@hook(priority=10)
def agent_prompt_prefix(prefix: str, cat) -> str:
    global custom_prefix
    prefix = custom_prefix
    return prefix


@hook(priority=10)
def agent_prompt_suffix(prompt_suffix: str, cat) -> str:
    global disable_memory
    if disable_memory:
        return ""

    return custom_suffix + """
 # Context
 {context}
 """


@hook(priority=1)
def before_cat_recalls_memories(config: RecallSettings, cat) -> RecallSettings:
    global disable_memory, number_of_memory_items, number_of_history_items, tags
    if disable_memory:
        custom_k = 1
        config.threshold = 1
    else:
        custom_k = number_of_memory_items
    config.k = custom_k
    config.threshold = threshold
    config.latest_n_history = number_of_history_items
    if tags:
        config.metadata |= tags

    return config


@hook(priority=1)
def agent_fast_reply(cat) -> AgenticWorkflowOutput | None:
    global only_local, custom_reply_no_memory
    if not only_local:
        return None

    num_memories = len(cat.working_memory.context_memories)
    if num_memories > 0:
        return None

    lang = get_message_language(cat.working_memory.user_message.text)

    translator = Translator(to_lang=lang)
    message = translator.translate(custom_reply_no_memory)

    return AgenticWorkflowOutput(output=message)


@hook
async def after_cat_recalls_memories(config: RecallSettings, cat) -> None:
    global metadata_or_filter, threshold, number_of_memory_items

    if not metadata_or_filter or not (tags_ := cat.working_memory.user_message.get("tags")):
        return

    config.metadata |= tags_
    await cat.recall_context_to_working_memory(config)
