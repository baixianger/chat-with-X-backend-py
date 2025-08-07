from langsmith import Client

client = Client()

# System prompts for main agent
ROUTER_SYSTEM_PROMPT = (
    client.pull_prompt("chat-with-x-router").messages[0].prompt.template
)


RESEARCH_PLAN_SYSTEM_PROMPT = (
    client.pull_prompt("chat-with-x-research-plan").messages[0].prompt.template
)


RESPONSE_SYSTEM_PROMPT = (
    client.pull_prompt("chat-with-x-response").messages[0].prompt.template
)

# System prompts for research agent after the main agent generate a research plan
GENERATE_QUERIES_SYSTEM_PROMPT = (
    client.pull_prompt("chat-with-x-queries").messages[0].prompt.template
)
