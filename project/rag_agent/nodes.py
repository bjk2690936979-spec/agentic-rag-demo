import json
import re
from typing import Any, Dict, List, Literal, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import BASE_TOKEN_THRESHOLD, MAX_ITERATIONS, MAX_TOOL_CALLS, TOKEN_GROWTH_FACTOR
from utils import estimate_context_tokens

from .graph_state import AgentState, State
from .prompts import *
from .schemas import QueryAnalysis


class FinalVerification(BaseModel):
    verified_answer: str = Field(description="Final answer to return after verification.")
    is_grounded: bool = Field(description="Whether the answer is sufficiently supported by the provided evidence.")
    issues: List[str] = Field(default_factory=list, description="Concrete grounding or evidence issues.")
    verdict: Literal["grounded", "weakly_grounded", "ungrounded"] = Field(
        description="Overall grounding verdict for the candidate answer."
    )
    used_conservative_rewrite: bool = Field(
        description="Whether the candidate answer was materially rewritten to be more conservative."
    )


def _summarize_text(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_payload(value: Any, limit: int = 200) -> Any:
    if isinstance(value, str):
        return _summarize_text(value, limit=limit)
    if isinstance(value, dict):
        return {k: _summarize_payload(v, limit=120) for k, v in value.items()}
    if isinstance(value, list):
        return [_summarize_payload(v, limit=120) for v in value[:5]]
    return value


def _extract_parent_ids(args: Dict[str, Any]) -> List[str]:
    raw = args.get("parent_id") or args.get("id") or args.get("ids") or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _get_trace_scope(question_index: Optional[int]) -> str:
    return "main" if question_index is None else f"agent[{question_index}]"


def _with_trace_metadata(
    state: Dict[str, Any],
    node_name: str,
    events: List[dict],
    reset: bool = False,
    base_offset: int = 0,
) -> List[dict]:
    question_index = state.get("question_index")
    base_index = (0 if reset else len(state.get("observability_events", []))) + base_offset
    scope = _get_trace_scope(question_index)

    traced_events = []
    for offset, event in enumerate(events, start=1):
        step_index = base_index + offset
        traced_events.append({
            "event_type": event["event_type"],
            "step_index": step_index,
            "node_name": node_name,
            "logical_order": f"{scope}:{step_index}",
            **event,
        })
    return traced_events


def _get_events(state: Dict[str, Any], event_type: str) -> List[Dict[str, Any]]:
    return [event for event in state.get("observability_events", []) if event.get("event_type") == event_type]


def _extract_source_names(content: Any) -> List[str]:
    matches = re.findall(r"(?m)^File Name:\s*(.+?)\s*$", str(content or ""))
    return _dedupe_preserve_order([match.strip() for match in matches if match.strip()])


def _extract_retrieval_enhancement_payload(content: Any) -> Optional[Dict[str, Any]]:
    match = re.search(
        r"\[RETRIEVAL_ENHANCEMENT\]\s*(\{.*?\})\s*\[/RETRIEVAL_ENHANCEMENT\]",
        str(content or ""),
        re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _strip_retrieval_enhancement_block(content: Any) -> str:
    return re.sub(
        r"\[RETRIEVAL_ENHANCEMENT\]\s*\{.*?\}\s*\[/RETRIEVAL_ENHANCEMENT\]\s*",
        "",
        str(content or ""),
        flags=re.DOTALL,
    ).strip()


def _compact_json(value: Any, limit: int = 3500) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    if len(payload) <= limit:
        return payload
    return payload[: limit - 3] + "..."


def _strip_sources_section(answer: str) -> str:
    if not answer:
        return ""
    parts = re.split(r"\n---\s*\n\*\*Sources:\*\*\s*\n", answer, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].rstrip() if parts else answer.strip()


def _infer_tool_success(result_content: str) -> bool:
    if result_content == "NO_TOOL_RESULT":
        return False
    return not (
        result_content.startswith("RETRIEVAL_ERROR:")
        or result_content.startswith("PARENT_RETRIEVAL_ERROR:")
    )


def _count_retrieval_hits(result_content: str) -> int:
    if result_content.startswith(("NO_RELEVANT_CHUNKS", "NO_PARENT_DOCUMENT", "NO_PARENT_DOCUMENTS")):
        return 0
    return len(re.findall(r"(?m)^Parent ID:\s*", result_content))


def _looks_like_conservative_answer(answer: str) -> bool:
    lowered = (answer or "").lower()
    markers = [
        "i couldn't find",
        "insufficient",
        "not enough information",
        "the available sources",
        "unable to determine",
        "cannot determine",
        "not supported by the retrieved",
    ]
    return any(marker in lowered for marker in markers)


def _build_evidence_bundle(state: State) -> Dict[str, Any]:
    retrieval_events = [
        event for event in state.get("observability_events", [])
        if event.get("event_type") == "tool_call"
    ]
    source_names = _dedupe_preserve_order([
        source
        for event in retrieval_events
        for source in event.get("source_names", [])
    ])
    parent_ids = _dedupe_preserve_order([
        parent_id
        for event in retrieval_events
        for parent_id in event.get("parent_ids", [])
    ])

    trace_items = []
    for event in retrieval_events[-12:]:
        trace_items.append({
            "logical_order": event.get("logical_order"),
            "tool_name": event.get("tool_name"),
            "success_flag": event.get("success_flag"),
            "retrieval_hit_count": event.get("retrieval_hit_count"),
            "source_names": event.get("source_names", [])[:5],
            "parent_ids": event.get("parent_ids", [])[:5],
            "result_summary": event.get("result_summary"),
        })

    return {
        "retrieval_event_count": len(retrieval_events),
        "source_names": source_names[:20],
        "parent_ids": parent_ids[:20],
        "trace_items": trace_items,
    }


def _collect_retrieval_guard_stats(state: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    enhancement_events = _get_events(state, "retrieval_enhancement")
    search_tool_events = [
        event for event in _get_events(state, "tool_call")
        if event.get("tool_name") == "search_child_chunks"
    ]

    total_query_count = sum(int(event.get("query_count", 0) or 0) for event in enhancement_events)
    total_raw_hit_count = sum(int(event.get("raw_hit_count", 0) or 0) for event in enhancement_events)
    total_deduped_hit_count = sum(int(event.get("deduped_hit_count", 0) or 0) for event in enhancement_events)
    max_deduped_hit_count = max(
        (int(event.get("deduped_hit_count", 0) or 0) for event in enhancement_events),
        default=0,
    )
    successful_search_count = sum(1 for event in search_tool_events if event.get("success_flag"))

    return {
        "enhancement_events": enhancement_events,
        "search_tool_events": search_tool_events,
        "query_count": total_query_count,
        "raw_hit_count": total_raw_hit_count,
        "deduped_hit_count": total_deduped_hit_count,
        "max_deduped_hit_count": max_deduped_hit_count,
        "successful_search_count": successful_search_count,
        "evidence_source_count": len(evidence_bundle.get("source_names", [])),
        "evidence_parent_count": len(evidence_bundle.get("parent_ids", [])),
    }


def _determine_robustness_guard(
    state: Dict[str, Any],
    verification: FinalVerification,
    evidence_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    fallback_events = _get_events(state, "fallback_response")
    fallback_reasons = [event.get("fallback_reason") for event in fallback_events]
    retrieval_stats = _collect_retrieval_guard_stats(state, evidence_bundle)

    if any(reason in {"budget_exhausted", "loop_limit_exceeded"} for reason in fallback_reasons):
        return {
            "downgrade_trigger": True,
            "downgrade_reason": "budget_or_loop_failure",
            "final_route": "fallback",
            "reason_message": "由于检索预算或循环次数限制，本轮只能输出保守结论。",
        }

    retrieval_insufficient = (
        bool(retrieval_stats["enhancement_events"]) and (
            retrieval_stats["raw_hit_count"] <= 0
            or retrieval_stats["deduped_hit_count"] <= 0
            or retrieval_stats["successful_search_count"] <= 0
            or (
                retrieval_stats["max_deduped_hit_count"] < 2
                and retrieval_stats["evidence_source_count"] < 2
            )
        )
    )
    if retrieval_insufficient:
        return {
            "downgrade_trigger": True,
            "downgrade_reason": "retrieval_insufficient",
            "final_route": "verified_conservative",
            "reason_message": "当前未检索到足够相关证据，无法支撑可靠回答。",
        }

    if verification.verdict == "ungrounded":
        return {
            "downgrade_trigger": True,
            "downgrade_reason": "verifier_rejected",
            "final_route": "verified_conservative",
            "reason_message": "候选答案缺少充分证据支持，已自动降级为更保守回答。",
        }

    if verification.verdict == "weakly_grounded" or not verification.is_grounded:
        return {
            "downgrade_trigger": True,
            "downgrade_reason": "evidence_weak",
            "final_route": "verified_conservative",
            "reason_message": "当前证据较弱或覆盖不完整，回答已按低置信口径处理。",
        }

    return {
        "downgrade_trigger": False,
        "downgrade_reason": "none",
        "final_route": "verified",
        "reason_message": "",
    }


def _apply_robustness_downgrade(
    answer_body: str,
    guard: Dict[str, Any],
    verification: FinalVerification,
    already_conservative: bool,
) -> Dict[str, Any]:
    if not guard["downgrade_trigger"]:
        return {
            "answer_body": answer_body,
            "used_conservative_rewrite": verification.used_conservative_rewrite,
        }

    downgrade_reason = guard["downgrade_reason"]
    if downgrade_reason == "retrieval_insufficient":
        downgraded = (
            answer_body if already_conservative else
            "当前未检索到足够相关证据，暂时不能给出可靠结论。\n\n"
            "基于现有检索结果，只能保守地说：目前证据不足以支持明确回答。"
        )
    elif downgrade_reason == "verifier_rejected":
        downgraded = (
            answer_body if (already_conservative or verification.used_conservative_rewrite) else
            "当前候选答案缺少充分证据支持，不能直接作为可靠结论。\n\n"
            "基于现有证据，只能保守地说：目前没有足够依据得出稳定结论。"
        )
    elif downgrade_reason == "evidence_weak":
        downgraded = (
            answer_body if already_conservative else
            f"基于当前有限证据，以下结论仅作低置信参考：\n\n{answer_body}"
        )
    elif downgrade_reason == "budget_or_loop_failure":
        downgraded = (
            answer_body if already_conservative else
            f"由于检索预算或循环次数限制，本轮研究提前收敛，以下内容按保守口径理解：\n\n{answer_body}"
        )
    else:
        downgraded = answer_body

    return {
        "answer_body": downgraded.strip(),
        "used_conservative_rewrite": True,
    }


def _build_validation_notes(
    verification: FinalVerification,
    evidence_bundle: Dict[str, Any],
    extra_notes: Optional[List[str]] = None,
) -> List[str]:
    notes: List[str] = []
    if extra_notes:
        notes.extend(extra_notes)
    if verification.verdict == "ungrounded":
        notes.append("以下内容缺少充分证据支持，已尽量按保守口径表达。")
    elif verification.verdict == "weakly_grounded":
        notes.append("部分内容证据较弱或覆盖不完整，请谨慎使用。")

    if verification.issues:
        notes.extend(verification.issues[:3])
    elif evidence_bundle.get("retrieval_event_count", 0) == 0:
        notes.append("当前回答未关联到有效检索证据。")

    return _dedupe_preserve_order(notes)


def _format_validation_block(
    verification: FinalVerification,
    evidence_bundle: Dict[str, Any],
    extra_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    source_names = evidence_bundle.get("source_names", [])[:6]
    parent_ids = evidence_bundle.get("parent_ids", [])[:6]
    notes = _build_validation_notes(verification, evidence_bundle, extra_notes=extra_notes)

    status_line = f"验证状态：`{verification.verdict}`"
    if not verification.is_grounded and verification.verdict != "grounded":
        status_line += "（低置信）"

    lines = [
        "---",
        "**验证信息**",
        f"- {status_line}",
    ]

    if source_names:
        lines.append(f"- 来源文件：{', '.join(source_names)}")
    if parent_ids:
        lines.append(f"- Parent IDs：{', '.join(parent_ids)}")
    if notes:
        lines.append(f"- 说明：{'；'.join(notes)}")

    return {
        "validation_block": "\n".join(lines),
        "display_source_names": source_names,
        "display_parent_ids": parent_ids,
        "display_notes": notes,
    }


def _build_fallback_reason_note(fallback_reason: str) -> str:
    reason_map = {
        "budget_exhausted": "失败原因：达到检索预算上限，本轮只能保守作答。",
        "loop_limit_exceeded": "失败原因：达到循环次数上限，本轮只能保守作答。",
        "no_retrieval_results": "失败原因：多 query 检索后仍未获得足够相关证据。",
        "other": "失败原因：当前证据链不完整，只能输出保守结论。",
    }
    reason_text = reason_map.get(fallback_reason, reason_map["other"])
    return f"**研究降级说明**\n- {reason_text}"


def _build_rewrite_event(last_message: HumanMessage, conversation_summary: str, response: QueryAnalysis) -> dict:
    return {
        "event_type": "rewrite_query",
        "input_query": _summarize_text(last_message.content, limit=180),
        "conversation_summary": _summarize_text(conversation_summary, limit=180),
        "is_clear": response.is_clear,
        "rewritten_questions": [_summarize_text(q, limit=140) for q in response.questions[:3]],
        "clarification_needed": _summarize_text(response.clarification_needed, limit=180),
    }


def _build_tool_events(state: AgentState) -> List[dict]:
    messages = state["messages"]
    latest_ai_index = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            latest_ai_index = idx
            break

    if latest_ai_index is None:
        return []

    tool_calls = messages[latest_ai_index].tool_calls
    tool_results = {}
    for msg in messages[latest_ai_index + 1:]:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg
        elif isinstance(msg, AIMessage):
            break

    events = []
    for call in tool_calls:
        args = call.get("args", {}) or {}
        result = tool_results.get(call.get("id"))
        result_content = getattr(result, "content", "NO_TOOL_RESULT")
        retrieval_payload = _extract_retrieval_enhancement_payload(result_content)
        result_body = _strip_retrieval_enhancement_block(result_content)
        source_names = (
            retrieval_payload.get("top_source_names", [])
            if retrieval_payload and retrieval_payload.get("top_source_names")
            else _extract_source_names(result_body)
        )
        event = {
            "event_type": "tool_call",
            "question_index": state.get("question_index", 0),
            "question": _summarize_text(state.get("question", ""), limit=160),
            "tool_name": call.get("name", "unknown"),
            "tool_args_summary": _summarize_payload(args),
            "result_summary": _summarize_text(result_body or result_content, limit=220),
            "success_flag": _infer_tool_success(str(result_content)),
            "retrieval_hit_count": (
                retrieval_payload.get("deduped_hit_count")
                if retrieval_payload and "deduped_hit_count" in retrieval_payload
                else _count_retrieval_hits(str(result_body or result_content))
            ),
            "source_names": source_names,
        }

        if call.get("name") == "retrieve_parent_chunks":
            event["parent_ids"] = _extract_parent_ids(args)
        elif call.get("name") == "search_child_chunks" and retrieval_payload:
            event["parent_ids"] = retrieval_payload.get("top_parent_ids", [])

        events.append(event)

        if call.get("name") == "search_child_chunks" and retrieval_payload:
            events.append({
                "event_type": "retrieval_enhancement",
                "question_index": state.get("question_index", 0),
                "question": _summarize_text(state.get("question", ""), limit=160),
                "original_query": retrieval_payload.get("original_query", args.get("query", "")),
                "expanded_queries": retrieval_payload.get("expanded_queries", []),
                "query_count": retrieval_payload.get("query_count", 1),
                "raw_hit_count": retrieval_payload.get("raw_hit_count", 0),
                "deduped_hit_count": retrieval_payload.get("deduped_hit_count", 0),
                "fusion_method": retrieval_payload.get("fusion_method", "unknown"),
                "top_source_names": retrieval_payload.get("top_source_names", []),
                "top_parent_ids": retrieval_payload.get("top_parent_ids", []),
            })

    return events


def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}


def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "")
        + f"User Query:\n{last_message.content}\n"
    )

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=context_section),
    ])
    rewrite_event = _with_trace_metadata(
        state,
        "rewrite_query",
        [_build_rewrite_event(last_message, conversation_summary, response)],
        reset=True,
    )

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
            "observability_events": [{"__reset__": True}] + rewrite_event,
        }

    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "I need more information to understand your question."
    )
    return {
        "questionIsClear": False,
        "messages": [AIMessage(content=clarification)],
        "observability_events": [{"__reset__": True}] + rewrite_event,
    }


def request_clarification(state: State):
    return {}


def verify_final_answer(state: State, llm):
    if not state.get("messages"):
        return {}

    candidate_message = state["messages"][-1]
    if not isinstance(candidate_message, AIMessage) or not candidate_message.content:
        return {}

    candidate_answer = candidate_message.content.strip()
    evidence_bundle = _build_evidence_bundle(state)
    retrieval_trace = evidence_bundle["trace_items"]
    already_conservative = _looks_like_conservative_answer(candidate_answer)

    verifier_input = HumanMessage(content=(
        f"Original Query:\n{state.get('originalQuery', '')}\n\n"
        f"Rewritten Queries:\n{_compact_json(state.get('rewrittenQuestions', []), limit=1200)}\n\n"
        f"Candidate Final Answer:\n{candidate_answer}\n\n"
        f"Already Conservative Style: {already_conservative}\n\n"
        f"Retrieval Trace:\n{_compact_json(retrieval_trace, limit=2200)}\n\n"
        f"Evidence Bundle:\n{_compact_json(evidence_bundle, limit=2200)}"
    ))

    llm_with_structure = llm.with_config(temperature=0).with_structured_output(FinalVerification)
    verification = llm_with_structure.invoke([
        SystemMessage(content=get_final_verifier_prompt()),
        verifier_input,
    ])

    verified_answer = verification.verified_answer.strip() if verification.verified_answer else candidate_answer
    if already_conservative and verification.verdict != "grounded" and not verification.used_conservative_rewrite:
        verified_answer = candidate_answer
    if not verified_answer:
        verified_answer = candidate_answer

    verified_answer_body = _strip_sources_section(verified_answer)
    guard = _determine_robustness_guard(state, verification, evidence_bundle)
    guarded_answer = _apply_robustness_downgrade(
        verified_answer_body,
        guard,
        verification,
        already_conservative,
    )
    final_answer_body = guarded_answer["answer_body"]
    extra_notes = [guard["reason_message"]] if guard["downgrade_trigger"] and guard["reason_message"] else []
    formatted_validation = _format_validation_block(
        verification,
        evidence_bundle,
        extra_notes=extra_notes,
    )
    final_answer = f"{final_answer_body}\n\n{formatted_validation['validation_block']}".strip()

    verification_event = _with_trace_metadata(state, "verify_final_answer", [{
        "event_type": "final_verification",
        "candidate_answer_summary": _summarize_text(candidate_answer, limit=220),
        "verified_answer_summary": _summarize_text(verified_answer, limit=220),
        "final_answer_summary": _summarize_text(final_answer, limit=260),
        "is_grounded": verification.is_grounded,
        "issues": verification.issues[:6],
        "verdict": verification.verdict,
        "used_conservative_rewrite": guarded_answer["used_conservative_rewrite"],
        "already_conservative_candidate": already_conservative,
        "evidence_source_names": evidence_bundle["source_names"],
        "evidence_parent_ids": evidence_bundle["parent_ids"],
        "retrieval_event_count": evidence_bundle["retrieval_event_count"],
        "has_validation_block": True,
        "display_source_names": formatted_validation["display_source_names"],
        "display_parent_ids": formatted_validation["display_parent_ids"],
        "display_notes": formatted_validation["display_notes"],
    }])
    guard_event = _with_trace_metadata(state, "verify_final_answer", [{
        "event_type": "robustness_guard",
        "downgrade_trigger": guard["downgrade_trigger"],
        "downgrade_reason": guard["downgrade_reason"],
        "original_verdict": verification.verdict,
        "final_route": guard["final_route"],
        "used_conservative_rewrite": guarded_answer["used_conservative_rewrite"],
    }], base_offset=len(verification_event))

    return {
        "messages": [RemoveMessage(id=candidate_message.id), AIMessage(content=final_answer)],
        "observability_events": verification_event + guard_event,
    }


# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,
    }


def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([
        SystemMessage(content=get_fallback_response_prompt()),
        HumanMessage(content=prompt_content),
    ])
    fallback_reason = _infer_fallback_reason(state)
    fallback_reason_note = _build_fallback_reason_note(fallback_reason)
    response_content = f"{fallback_reason_note}\n\n{response.content}".strip()

    return {
        "messages": [AIMessage(content=response_content)],
        "response_route": "fallback",
        "observability_events": _with_trace_metadata(state, "fallback_response", [{
            "event_type": "fallback_response",
            "question_index": state.get("question_index", 0),
            "question": _summarize_text(state.get("question", ""), limit=160),
            "fallback_reason": fallback_reason,
        }]),
    }


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids
    tool_events = _build_tool_events(state)

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    tool_events = _with_trace_metadata(state, "tools", tool_events)
    compression_event = _with_trace_metadata(state, "should_compress_context", [{
        "event_type": "context_compression_check",
        "question_index": state.get("question_index", 0),
        "question": _summarize_text(state.get("question", ""), limit=160),
        "triggered": goto == "compress_context",
        "current_tokens": current_tokens,
        "max_allowed_tokens": max_allowed,
    }], reset=False, base_offset=len(tool_events))
    return Command(
        update={
            "retrieval_keys": updated_ids,
            "observability_events": tool_events + compression_event,
        },
        goto=goto,
    )


def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT - {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([
        SystemMessage(content=get_context_compression_prompt()),
        HumanMessage(content=conversation_text),
    ])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    final_route = state.get("response_route") or "normal"
    grounded_candidate = _infer_grounded_candidate(state, answer)
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}],
        "observability_events": _with_trace_metadata(state, "collect_answer", [{
            "event_type": "final_response",
            "question_index": state.get("question_index", 0),
            "question": _summarize_text(state.get("question", ""), limit=160),
            "route": final_route,
            "answer_summary": _summarize_text(answer, limit=220),
            "grounded_candidate": grounded_candidate,
        }]),
    }


# --- End of Agent Nodes---
def _infer_fallback_reason(state: AgentState) -> str:
    if state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "budget_exhausted"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "loop_limit_exceeded"

    tool_events = [event for event in state.get("observability_events", []) if event.get("event_type") == "tool_call"]
    if tool_events and not any((event.get("retrieval_hit_count") or 0) > 0 for event in tool_events):
        return "no_retrieval_results"

    return "other"


def _infer_grounded_candidate(state: AgentState, answer: str) -> Any:
    if not answer or answer == "Unable to generate an answer.":
        return False

    tool_events = [event for event in state.get("observability_events", []) if event.get("event_type") == "tool_call"]
    if not tool_events:
        return "unknown"

    if any(event.get("success_flag") and (event.get("retrieval_hit_count") or 0) > 0 for event in tool_events):
        return True

    return False


def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n" f"{ans['answer']}\n")

    user_message = HumanMessage(
        content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}"""
    )
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}
