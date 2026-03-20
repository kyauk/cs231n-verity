"""
Multi-agent debate workflow using LangGraph + NVIDIA NIM.

Stage 1: Generate 20 synthetic AV regression scenarios.
Stage 2: Run a multi-round Proponent <-> Critic debate about whether a severe
         outlier scenario should be added to the regression suite.
Stage 3: Judge outputs strict binary decision: yes/no.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class DebateState(TypedDict):
    """
    Purpose: Shared state passed through the LangGraph debate workflow.
    Parameters:
        description (str): Optional human-readable motion summary.
        regression_suite (list[str]): Baseline synthetic regression scenarios.
        candidate_scenario (str): Outlier scenario under debate.
        proponent_arg (str): Latest proponent turn.
        critic_arg (str): Latest critic turn.
        debate_history (list[str]): Ordered debate transcript.
        round_number (int): Current round index (1-indexed).
        max_rounds (int): Total rounds to run before judging.
        final_decision (str): Final binary verdict ("yes" or "no").
    Returns:
        TypedDict: Mutable workflow state for LangGraph.
    Called by: Graph runtime during app.invoke()
    Calls: None
    """

    description: str
    regression_suite: list[str]
    candidate_scenario: str
    proponent_arg: str
    critic_arg: str
    debate_history: list[str]
    round_number: int
    max_rounds: int
    final_decision: str


class JudgeResult(BaseModel):
    """
    Purpose: Enforce strict binary judge output.
    Parameters:
        decision (Literal["yes", "no"]): Final decision token.
    Returns:
        BaseModel: Parsed structured judge response.
    Called by: Judge_Node()
    Calls: None
    """

    decision: Literal["yes", "no"] = Field(
        ...,
        description="Return exactly 'yes' or 'no'.",
    )


class RegressionSuiteResult(BaseModel):
    """
    Purpose: Structured synthetic regression suite generation response.
    Parameters:
        scenarios (list[str]): Exactly 20 diverse AV regression scenarios.
    Returns:
        BaseModel: Parsed suite generation output.
    Called by: generate_regression_suite()
    Calls: None
    """

    scenarios: list[str] = Field(
        ...,
        description="Exactly 20 concise regression scenario descriptions for AV systems.",
    )


def _build_model() -> ChatNVIDIA:
    """
    Purpose: Create the shared NVIDIA NIM chat model used by all stages.
    Parameters:
        None
    Returns:
        ChatNVIDIA: Configured model client.
    Called by: generate_regression_suite(), Proponent_Node(), Critic_Node(), Judge_Node()
    Calls: ChatNVIDIA()
    """

    return ChatNVIDIA(model="meta/llama-3.1-70b-instruct")


def _format_suite(scenarios: list[str]) -> str:
    """
    Purpose: Render suite scenarios into a numbered multiline string for prompts.
    Parameters:
        scenarios (list[str]): Regression scenario list.
    Returns:
        str: Numbered suite text block.
    Called by: Proponent_Node(), Critic_Node(), Judge_Node(), print_debate_transcript()
    Calls: None
    """

    return "\n".join([f"{index + 1}. {item}" for index, item in enumerate(scenarios)])


def generate_regression_suite() -> list[str]:
    """
    Purpose: Synthetically generate 20 common AV regression test scenarios.
    Parameters:
        None
    Returns:
        list[str]: Exactly 20 concise scenario descriptions.
    Called by: main()
    Calls: _build_model(), ChatNVIDIA.with_structured_output(), invoke()
    """

    llm = _build_model().with_structured_output(RegressionSuiteResult)
    messages = [
        SystemMessage(
            content=(
                "You generate AUTONOMOUS VEHICLE (self-driving car) regression test suites.\n"
                "Return exactly 20 common but diverse autonomous-driving regression scenarios.\n"
                "Each scenario should be one concise sentence.\n"
                "Do not generate antivirus/software-security scenarios."
            )
        ),
        HumanMessage(content="Generate 20 common autonomous vehicle regression test scenarios."),
    ]
    result = llm.invoke(messages)

    scenarios: list[str] = []
    if isinstance(result, RegressionSuiteResult):
        scenarios = result.scenarios
    elif isinstance(result, dict):
        raw = result.get("scenarios")
        if isinstance(raw, list):
            scenarios = [str(item).strip() for item in raw if str(item).strip()]

    # If structured parsing yields nothing useful, fallback to plain-text parsing.
    if len(scenarios) == 0:
        fallback_response = _build_model().invoke(
            [
                SystemMessage(
                    content=(
                        "Generate exactly 20 common autonomous vehicle regression test scenarios. "
                        "Do not generate antivirus scenarios. "
                        "Output as a numbered list, one scenario per line."
                    )
                ),
                HumanMessage(content="Give me 20 common autonomous vehicle regression test scenarios."),
            ]
        )
        fallback_text = str(fallback_response.content or "")
        parsed_lines = []
        for raw_line in fallback_text.splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[\).\:-]|[-*])\s*", "", raw_line).strip()
            if cleaned and not cleaned.lower().startswith("here are 20"):
                parsed_lines.append(cleaned)
        scenarios = parsed_lines

    # Keep exactly 20 items for deterministic downstream behavior.
    scenarios = scenarios[:20]
    while len(scenarios) < 20:
        scenarios.append(f"Synthetic fallback scenario {len(scenarios) + 1}")
    return scenarios


def Proponent_Node(state: DebateState) -> DebateState:
    """
    Purpose: Argue that the candidate outlier should be added to regression tests.
    Parameters:
        state (DebateState): Current debate state.
    Returns:
        DebateState: Updated proponent_arg and debate_history.
    Called by: LangGraph node execution
    Calls: _build_model(), _format_suite(), ChatNVIDIA.invoke()
    """

    llm = _build_model()
    latest_critic = state["critic_arg"] or "No critic argument yet."
    suite_text = _format_suite(state["regression_suite"])
    messages = [
        SystemMessage(
            content=(
                "You are Proponent_Node in an AV safety review debate.\n"
                "Argue in exactly one paragraph that the candidate scenario SHOULD be added "
                "to the regression suite, using severity and gap analysis vs existing suite."
            )
        ),
        HumanMessage(
            content=(
                f"Round: {state['round_number']} of {state['max_rounds']}\n\n"
                f"Candidate scenario under debate:\n{state['candidate_scenario']}\n\n"
                f"Existing regression suite:\n{suite_text}\n\n"
                f"Latest critic argument:\n{latest_critic}"
            )
        ),
    ]
    response = llm.invoke(messages)
    proponent_arg = str(response.content).strip()
    updated_history = state["debate_history"] + [
        f"Round {state['round_number']} - Proponent: {proponent_arg}"
    ]
    return {
        "proponent_arg": proponent_arg,
        "debate_history": updated_history,
    }


def Critic_Node(state: DebateState) -> DebateState:
    """
    Purpose: Argue against adding candidate scenario (redundancy, feasibility, or noise).
    Parameters:
        state (DebateState): Current debate state.
    Returns:
        DebateState: Updated critic_arg, debate_history, and round_number.
    Called by: LangGraph node execution
    Calls: _build_model(), _format_suite(), ChatNVIDIA.invoke()
    """

    llm = _build_model()
    suite_text = _format_suite(state["regression_suite"])
    messages = [
        SystemMessage(
            content=(
                "You are Critic_Node in an AV safety review debate.\n"
                "In exactly one paragraph, rebut the proponent and argue why the candidate "
                "should NOT be added now (e.g., redundancy with suite, low incremental value, "
                "or flawed prioritization). Be rigorous and skeptical."
            )
        ),
        HumanMessage(
            content=(
                f"Round: {state['round_number']} of {state['max_rounds']}\n\n"
                f"Candidate scenario:\n{state['candidate_scenario']}\n\n"
                f"Existing regression suite:\n{suite_text}\n\n"
                f"Latest proponent argument:\n{state['proponent_arg']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    critic_arg = str(response.content).strip()
    updated_history = state["debate_history"] + [
        f"Round {state['round_number']} - Critic: {critic_arg}"
    ]
    return {
        "critic_arg": critic_arg,
        "debate_history": updated_history,
        "round_number": state["round_number"] + 1,
    }


def Judge_Node(state: DebateState) -> DebateState:
    """
    Purpose: Decide if candidate scenario should be added to suite (yes/no).
    Parameters:
        state (DebateState): Completed debate state with transcript.
    Returns:
        DebateState: Updated final_decision.
    Called by: LangGraph node execution after loop termination
    Calls: _build_model(), _format_suite(), ChatNVIDIA.with_structured_output(), invoke()
    """

    judge_llm = _build_model().with_structured_output(JudgeResult)
    transcript = "\n".join(state["debate_history"]) if state["debate_history"] else "No transcript."
    suite_text = _format_suite(state["regression_suite"])
    messages = [
        SystemMessage(
            content=(
                "You are Judge_Node for AV regression planning.\n"
                "Given existing suite and debate transcript, decide if candidate scenario "
                "should be added now.\n"
                "Return strictly structured decision 'yes' or 'no'."
            )
        ),
        HumanMessage(
            content=(
                f"Candidate scenario:\n{state['candidate_scenario']}\n\n"
                f"Existing regression suite:\n{suite_text}\n\n"
                f"Debate transcript:\n{transcript}"
            )
        ),
    ]
    result = judge_llm.invoke(messages)
    decision: str = "no"

    if isinstance(result, JudgeResult):
        decision = result.decision
    elif isinstance(result, dict):
        raw_decision = result.get("decision")
        if raw_decision in {"yes", "no"}:
            decision = raw_decision

    return {"final_decision": "yes" if decision == "yes" else "no"}


def _next_after_critic(state: DebateState) -> Literal["Proponent_Node", "Judge_Node"]:
    """
    Purpose: Route either to another round or to judge after max rounds reached.
    Parameters:
        state (DebateState): Current loop state.
    Returns:
        Literal["Proponent_Node", "Judge_Node"]: Next node name.
    Called by: LangGraph conditional edge
    Calls: None
    """

    if state["round_number"] <= state["max_rounds"]:
        return "Proponent_Node"
    return "Judge_Node"


def build_debate_graph():
    """
    Purpose: Build and compile looping Proponent/Critic debate graph with Judge terminal.
    Parameters:
        None
    Returns:
        CompiledStateGraph: Executable graph app.
    Called by: main()
    Calls: StateGraph(), add_node(), add_edge(), add_conditional_edges(), compile()
    """

    graph = StateGraph(DebateState)
    graph.add_node("Proponent_Node", Proponent_Node)
    graph.add_node("Critic_Node", Critic_Node)
    graph.add_node("Judge_Node", Judge_Node)

    graph.add_edge(START, "Proponent_Node")
    graph.add_edge("Proponent_Node", "Critic_Node")
    graph.add_conditional_edges("Critic_Node", _next_after_critic)
    graph.add_edge("Judge_Node", END)

    return graph.compile()


def print_debate_transcript(result_state: DebateState) -> None:
    """
    Purpose: Print generated suite, candidate scenario, transcript, and verdict.
    Parameters:
        result_state (DebateState): Final workflow state.
    Returns:
        None
    Called by: main()
    Calls: _format_suite(), print()
    """

    print("\n=== SYNTHETIC REGRESSION SUITE (20) ===")
    print(_format_suite(result_state["regression_suite"]))
    print("\n=== CANDIDATE OUTLIER SCENARIO ===")
    print(result_state["candidate_scenario"])
    print("\n=== DEBATE TRANSCRIPT ===")
    for turn in result_state["debate_history"]:
        print(f"- {turn}")
    print("\n=== JUDGE VERDICT ===")
    print(result_state["final_decision"])


def main() -> None:
    """
    Purpose: Run full synthetic-suite + multi-agent debate + binary verdict workflow.
    Parameters:
        None
    Returns:
        None
    Called by: __main__
    Calls: generate_regression_suite(), build_debate_graph(), app.invoke(), print_debate_transcript()
    """

    generated_suite = generate_regression_suite()
    candidate = (
        "Pedestrian walking at midnight during heavy rain with no reflective gear, "
        "low street lighting, and reduced sensor visibility."
    )

    app = build_debate_graph()
    initial_state: DebateState = {
        "description": "Debate whether the candidate should be added to regression suite.",
        "regression_suite": generated_suite,
        "candidate_scenario": candidate,
        "proponent_arg": "",
        "critic_arg": "",
        "debate_history": [],
        "round_number": 1,
        "max_rounds": 3,
        "final_decision": "",
    }

    result = app.invoke(initial_state)
    print_debate_transcript(result)

    print("\n=== BINARY OUTPUT ===")
    print(result["final_decision"])


if __name__ == "__main__":
    main()
