"""
Retrieval Graph.
"""
# pylint: disable=wrong-import-position
import os
import sys
from typing import cast, Literal, Any

# Add the project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import all modules after setting up sys.path
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from src.utils import format_docs, load_chat_model
from src.configuration import Configuration
from src.vectorstore import get_collection_list
from src.agent.state import AgentState, RouterState, Plan, InputState
from src.agent.researcher.graph import graph as researcher_graph
from src.agent.researcher.state import ResearcherState




async def analyze_and_route_query(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[str] | str | None]:
    """Analyze the user's query and determine the appropriate routing."""
    if state.type and state.logic:  # for testing
        return {"type": state.type, "logic": state.logic}
    configuration = Configuration.model_validate(config.get("configurable"))
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    llm, _, _ = load_chat_model(configuration.query_model)
    model = llm.with_structured_output(RouterState, include_raw=False, **structured_output_kwargs)
    messages = [
        {
            "role": "system",
            "content": configuration.router_system_prompt.format(
                collection_list=get_collection_list(configuration.retriever_provider)
            ),
        }
    ] + state.messages
    router = cast(RouterState, await model.ainvoke(messages))
    return {
        "type": router.type,
        "collections": router.collections,
        "response": router.response,
    }


def route_query(
    state: AgentState,
) -> Literal[
    "create_research_plan",
    "ask_for_more_info",
    "respond_to_general_query",
    "respond_to_unrelated_query",
]:
    """Determine the next step based on the query classification."""
    _type = state.type
    if _type == "more-info":
        return "ask_for_more_info"
    elif _type == "related":
        return "create_research_plan"
    elif _type == "chitchat":
        return "respond_to_general_query"
    elif _type == "unrelated":
        return "respond_to_unrelated_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_more_info(
    state: AgentState,
) -> dict[str, list[BaseMessage]]:
    """Ask for more information from the user."""
    if not state.response:
        raise ValueError("Router response is empty, when more information is needed")
    message = AIMessage(content=state.response)
    return {"messages": [message]}


async def respond_to_general_query(
    state: AgentState,
) -> dict[str, list[BaseMessage]]:
    """Respond to a general question."""
    if not state.response:
        raise ValueError("Router response is empty, when answer to a general question")
    message = AIMessage(content=state.response)
    return {"messages": [message]}


async def respond_to_unrelated_query(
    state: AgentState,
) -> dict[str, list[BaseMessage]]:
    """Respond to an unrelated question."""
    if not state.response:
        raise ValueError(
            "Router response is empty, when answer to an unrelated question"
        )
    message = AIMessage(content=state.response)
    return {"messages": [message]}


async def create_research_plan(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[str] | str | Any]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    configuration = Configuration.model_validate(config.get("configurable"))
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model, _, _ = load_chat_model(configuration.query_model)
    model = model.with_structured_output(Plan, include_raw=False, **structured_output_kwargs)
    messages = [
        {"role": "system", "content": configuration.research_plan_system_prompt}
    ] + state.messages
    research_plan = cast(
        Plan, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )
    return {
        "steps": research_plan.steps,
        "documents": "delete",
        "question": state.messages[-1].content,
        # keep the question always pointing to the latest user message
    }


async def conduct_research(
    state: AgentState,
) -> dict[str, list[Document] | list[str] | str] | AgentState:
    """Conduct research based on the research plan."""

    research_input = ResearcherState(
        step=state.steps[0],
        collections=state.collections or []
    )
    step_result = await researcher_graph.ainvoke(research_input)
    return {
        "documents": step_result["documents"],
        "steps": state.steps[1:],
        # remove the step that has been executed
    }


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Check if the research is finished."""
    if len(state.steps or []) > 0:
        return "conduct_research"  # steps 不为空 说明还有没执行完的step
    else:
        return "respond"  # 为空说明执行完毕


async def respond(
    state: AgentState, config: RunnableConfig
) -> dict[str, list[BaseMessage] | str | Any]:
    """Respond to the retrieved documents and the user's question."""
    configuration = Configuration.model_validate(config.get("configurable"))
    model, _, _ = load_chat_model(configuration.response_model)
    # add a re-ranker here, todo
    top_k = 20
    context = format_docs(state.documents[:top_k])
    prompt = configuration.response_system_prompt.format(
        collections=state.collections, context=context
    )
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response], "answer": response.content}


builder = StateGraph(
    state_schema=AgentState,
    input_schema=InputState,
    context_schema=Configuration,
)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_unrelated_query)
builder.add_node(respond_to_general_query)
builder.add_node(create_research_plan)
builder.add_node(conduct_research)
builder.add_node(respond)


builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", path=route_query)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond_to_unrelated_query", END)
builder.add_edge("ask_for_more_info", "analyze_and_route_query")
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"


if __name__ == "__main__":

    input_state = InputState(messages=[HumanMessage(content="How to build a chatbot?")])
    agent_config = RunnableConfig(configurable={
        "retriever_provider": "chroma", 
        "storage_type": "persistent",
    })
    import asyncio
    result = asyncio.run(graph.ainvoke(input_state, agent_config))
    print(result)
