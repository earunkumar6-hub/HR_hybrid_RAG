import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from utils import api_client, graph_viz  # noqa: E402

st.set_page_config(
    page_title="Internal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role", "content", "sources", "entity_context", "relationship_context"}]


# ----------------------------------------------------------------------
# Sidebar — deliberately minimal: there's no department filter (this app
# doesn't have departments) and no upload control (documents are read
# straight from backend/data/documents/ and indexed automatically when
# the backend starts — see that folder's README.md).
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("HR Internal Knowledge Assistant")

    try:
        health = api_client.health()
        st.success("Backend connected")
        with st.expander("Backend status", expanded=False):
            st.json(health)
    except Exception as exc:
        st.error(f"Backend unreachable: {exc}")
        st.caption(f"Expected at `{api_client.BACKEND_URL}` — start it with `uvicorn app:app` in `backend/`.")

    st.divider()
    use_kg = st.toggle(
        "Use knowledge-graph context",
        value=True,
        help="Adds related entities (EE) and relationships (RM) from the knowledge graph to the answer's context, on top of the retrieved document excerpts (DL).",
    )

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_chat, tab_graph, tab_eval = st.tabs(["💬 Chat", "🕸️ Knowledge Graph", "📊 Evaluation"])

# ---- Chat tab ----------------------------------------------------------
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("blocked"):
                    st.warning(f"Blocked by guardrail ({msg.get('engine', 'unknown')} engine)")
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} document excerpt(s)"):
                        for s in msg["sources"]:
                            st.markdown(f"**{s['filename']}** · chunk #{s['chunk_index']} · score={s['score']}")
                            st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
                if msg.get("entity_context"):
                    with st.expander(f"🕸️ {len(msg['entity_context'])} related entities"):
                        for e in msg["entity_context"]:
                            related = ", ".join(e["related_to"]) if e["related_to"] else "—"
                            st.markdown(f"- **{e['entity']}** → {related}")
                if msg.get("relationship_context"):
                    with st.expander(f"🔗 {len(msg['relationship_context'])} related relationships"):
                        for r in msg["relationship_context"]:
                            st.markdown(f"- {r['source']} — *{r['relation']}* — {r['target']}")

    question = st.chat_input("Ask about anything in the indexed knowledge base...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    resp = api_client.ask(question, chat_history=history_payload, use_knowledge_graph=use_kg)
                    st.markdown(resp["answer"])
                    if resp.get("blocked"):
                        st.warning(f"Blocked by guardrail ({resp.get('guardrail_engine', 'unknown')} engine)")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": resp["answer"],
                            "sources": resp.get("sources", []),
                            "entity_context": resp.get("entity_context", []),
                            "relationship_context": resp.get("relationship_context", []),
                            "blocked": resp.get("blocked", False),
                            "engine": resp.get("guardrail_engine"),
                        }
                    )
                except Exception as exc:
                    error_msg = f"Request failed: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---- Knowledge graph tab -----------------------------------------------
with tab_graph:
    st.subheader("Knowledge graph")
    st.caption(
        "⬛ document nodes · ⚫ entity nodes, coloured by entity type "
        "(person / team / system / policy / process / project / regulation)."
    )
    if st.button("Refresh graph"):
        st.rerun()

    try:
        graph_data = api_client.get_graph()
        stats = graph_data.get("stats", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total nodes", stats.get("total_nodes", 0))
        c2.metric("Total edges", stats.get("total_edges", 0))
        c3.metric("Shown", stats.get("shown_nodes", 0))

        if graph_data.get("nodes"):
            html = graph_viz.render_graph_html(graph_data)
            st.components.v1.html(html, height=650, scrolling=True)
        else:
            st.info(
                "No graph data yet — drop files into `backend/data/documents/` "
                "and (re)start the backend to index them."
            )
    except Exception as exc:
        st.error(f"Could not load graph: {exc}")

# ---- Evaluation tab -----------------------------------------------------
with tab_eval:
    st.subheader("DeepEval — evaluate a chat answer")
    st.caption(
        "Scores the most recent assistant answer with reference-free RAG "
        "metrics (Answer Relevancy, Faithfulness, Contextual Relevancy). "
        "Add an optional expected/ground-truth answer to also compute "
        "Contextual Precision & Recall."
    )

    assistant_turns = [m for m in st.session_state.messages if m["role"] == "assistant" and m.get("sources") is not None]
    if not assistant_turns:
        st.info("Ask a question in the Chat tab first, then come back here to evaluate the answer.")
    else:
        options = {f"Turn {i+1}: {m['content'][:60]}...": m for i, m in enumerate(assistant_turns)}
        choice_key = st.selectbox("Choose an answer to evaluate", options=list(options.keys()))
        chosen = options[choice_key]
        question_lookup_idx = st.session_state.messages.index(chosen) - 1
        question_text = (
            st.session_state.messages[question_lookup_idx]["content"]
            if question_lookup_idx >= 0
            else ""
        )

        expected = st.text_area("Expected / ground-truth answer (optional)")
        if st.button("Run evaluation"):
            with st.spinner("Running DeepEval metrics (LLM-as-judge — usually 15-40s, can take longer)..."):
                try:
                    context_texts = [s["text"] for s in chosen.get("sources", [])]
                    result = api_client.evaluate(
                        question=question_text,
                        answer=chosen["content"],
                        retrieved_context=context_texts,
                        expected_output=expected or None,
                    )
                    cols = st.columns(len(result["metrics"]))
                    for col, m in zip(cols, result["metrics"]):
                        col.metric(
                            m["name"].replace("Metric", ""),
                            f"{m['score']:.2f}",
                            delta="✅ pass" if m["passed"] else "❌ fail",
                        )
                    with st.expander("Details"):
                        st.json(result)
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")
