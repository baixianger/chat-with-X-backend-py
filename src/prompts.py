from langsmith import Client

client = Client()

# System prompts for main agent
ROUTER_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-router-prompt")
    .messages[0]
    .prompt.template
)

MORE_INFO_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-more-info-prompt")
    .messages[0]
    .prompt.template
)

RESEARCH_PLAN_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-research-plan-prompt")
    .messages[0]
    .prompt.template
)

GENERAL_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-general-prompt")
    .messages[0]
    .prompt.template
)

RESPONSE_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-response-prompt")
    .messages[0]
    .prompt.template
)

# System prompts for research agent after the main agent generate a research plan
GENERATE_QUERIES_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-generate-queries-prompt")
    .messages[0]
    .prompt.template
)
