from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    build_search_plan_node,
    general_answer_node,
    paper_search_node,
    planner_node,
    research_answer_node,
    route_question,
)
from agent.state import AgentState


def create_agent_graph():
    builder = StateGraph(AgentState)

    # 1. 注册节点
    builder.add_node(
        "planner",
        planner_node,
    )

    builder.add_node(
        "general_answer",
        general_answer_node,
    )

    builder.add_node(
        "build_search_plan",
        build_search_plan_node,
    )

    builder.add_node(
        "paper_search",
        paper_search_node,
    )

    builder.add_node(
        "research_answer",
        research_answer_node,
    )

    # 2. 图的入口
    builder.add_edge(
        START,
        "planner",
    )

    # 3. Planner 条件路由
    builder.add_conditional_edges(
        "planner",
        route_question,
        {
            "general": "general_answer",
            "research": "build_search_plan",
        },
    )

    # 4. Research 路线
    builder.add_edge(
        "build_search_plan",
        "paper_search",
    )

    builder.add_edge(
        "paper_search",
        "research_answer",
    )

    # 5. 两条路线的结束节点
    builder.add_edge(
        "general_answer",
        END,
    )

    builder.add_edge(
        "research_answer",
        END,
    )

    return builder.compile()


agent_graph = create_agent_graph()


def run_agent(question: str) -> str:
    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    result = agent_graph.invoke(
        {
            "question": question,
        }
    )

    answer = result.get("answer")

    if not answer:
        raise RuntimeError(
            "Agent finished without producing an answer."
        )

    return answer