from translate import Translator
from langdetect import detect, DetectorFactory

from cat import hook, AgenticWorkflowOutput, RecallSettings, StrayCat


def get_message_language(message):
    # to guarantee consistent results (recommended)
    DetectorFactory.seed = 0
    try:
        lang = detect(message)
        return lang  # es. 'it', 'en', 'fr'
    except:
        # Fallback
        return "en"


@hook(priority=10)
async def agent_prompt_prefix(prefix: str, cat: StrayCat) -> str:
    settings = await cat.plugin_manager.get_plugin().load_settings()

    prefix = settings["prompt_prefix"]
    return prefix


@hook(priority=10)
async def agent_prompt_suffix(prompt_suffix: str, cat: StrayCat) -> str:
    settings = await cat.plugin_manager.get_plugin().load_settings()
    if settings["disable_memories"]:
        return ""

    return settings["prompt_suffix"] + """
 # Context
 {context}
 """


@hook(priority=1)
async def before_cat_recalls_memories(config: RecallSettings, cat: StrayCat) -> RecallSettings:
    settings = await cat.plugin_manager.get_plugin().load_settings()

    if settings["disable_memories"]:
        custom_k = 1
        config.threshold = 1
    else:
        custom_k = settings["number_of_memory_items"]
    config.k = custom_k
    config.threshold = settings["threshold"]
    config.latest_n_history = settings["number_of_history_items"]

    metadata_or_filter = settings["enable_OR_condition_for_metadata_filter"]

    if metadata_or_filter and (tags := cat.working_memory.user_message.get("tags")):
        config.metadata |= tags

    return config


@hook(priority=1)
async def agent_fast_reply(cat: StrayCat) -> AgenticWorkflowOutput | None:
    settings = await cat.plugin_manager.get_plugin().load_settings()

    if not settings["only_local_responses"]:
        return None

    num_memories = len(cat.working_memory.context_memories)
    if num_memories > 0:
        return None

    lang = get_message_language(cat.working_memory.user_message.text)

    translator = Translator(to_lang=lang)
    message = translator.translate(settings["reply_no_memory"])

    return AgenticWorkflowOutput(output=message)
