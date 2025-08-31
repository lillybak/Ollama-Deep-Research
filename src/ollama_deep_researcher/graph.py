import json

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import (
    deduplicate_and_format_sources,
    tavily_search,
    format_sources,
    perplexity_search,
    duckduckgo_search,
    searxng_search,
    strip_thinking_tokens,
    get_config_value,
)
from ollama_deep_researcher.rag import create_rag_system
from ollama_deep_researcher.state import (
    SummaryState,
    SummaryStateInput,
    SummaryStateOutput,
)
from ollama_deep_researcher.prompts import (
    query_writer_instructions,
    summarizer_instructions,
    reflection_instructions,
    get_current_date,
    json_mode_query_instructions,
    tool_calling_query_instructions,
    json_mode_reflection_instructions,
    tool_calling_reflection_instructions,
)
from ollama_deep_researcher.lmstudio import ChatLMStudio

# Constants
MAX_TOKENS_PER_SOURCE = 1000
CHARS_PER_TOKEN = 4

def generate_search_query_with_structured_output(
    configurable: Configuration,
    messages: list,
    tool_class,
    fallback_query: str,
    tool_query_field: str,
    json_query_field: str,
):
    """Helper function to generate search queries using either tool calling or JSON mode.
    
    Args:
        configurable: Configuration object
        messages: List of messages to send to LLM
        tool_class: Tool class for tool calling mode
        fallback_query: Fallback search query if extraction fails
        tool_query_field: Field name in tool args containing the query
        json_query_field: Field name in JSON response containing the query
        
    Returns:
        Dictionary with "search_query" key
    """
    if configurable.use_tool_calling:
        llm = get_llm(configurable).bind_tools([tool_class])
        result = llm.invoke(messages)

        if not result.tool_calls:
            return {"search_query": fallback_query}
        
        try:
            tool_data = result.tool_calls[0]["args"]
            search_query = tool_data.get(tool_query_field)
            return {"search_query": search_query}
        except (IndexError, KeyError):
            return {"search_query": fallback_query}
    
    else:
        # Use JSON mode
        llm = get_llm(configurable)
        result = llm.invoke(messages)
        print(f"result: {result}")
        content = result.content

        try:
            parsed_json = json.loads(content)
            search_query = parsed_json.get(json_query_field)
            if not search_query:
                return {"search_query": fallback_query}
            return {"search_query": search_query}
        except (json.JSONDecodeError, KeyError):
            if configurable.strip_thinking_tokens:
                content = strip_thinking_tokens(content)
            return {"search_query": fallback_query}

def get_llm(configurable: Configuration):
    """Helper function to initialize LLM based on configuration.

    Uses JSON mode if use_tool_calling is False, otherwise regular mode for tool calling.

    Args:
        configurable: Configuration object containing LLM settings

    Returns:
        Configured LLM instance
    """
    if configurable.llm_provider == "lmstudio":
        if configurable.use_tool_calling:
            return ChatLMStudio(
                base_url=configurable.lmstudio_base_url,
                model=configurable.local_llm,
                temperature=0,
            )
        else:
            return ChatLMStudio(
                base_url=configurable.lmstudio_base_url,
                model=configurable.local_llm,
                temperature=0,
                format="json",
            )
    else:  # Default to Ollama
        if configurable.use_tool_calling:
            return ChatOllama(
                base_url=configurable.ollama_base_url,
                model=configurable.local_llm,
                temperature=0,
            )
        else:
            return ChatOllama(
                base_url=configurable.ollama_base_url,
                model=configurable.local_llm,
                temperature=0,
                format="json",
            )

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.

    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.

    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date, research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)

    @tool
    class Query(BaseModel):
        """
        This tool is used to generate a query for web search.
        """

        query: str = Field(description="The actual search query string")
        rationale: str = Field(
            description="Brief explanation of why this query is relevant"
        )

    messages = [
        SystemMessage(
            content=formatted_prompt + (
                tool_calling_query_instructions if configurable.use_tool_calling 
                else json_mode_query_instructions
            )
        ),
        HumanMessage(content="Generate a query for web search:"),
    ]

    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=Query,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="query",
        json_query_field="query",
    )


def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.

    Executes a web search using the configured search API (tavily, perplexity,
    duckduckgo, or searxng) and formats the results for further processing.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query,
            fetch_full_page=configurable.fetch_full_page,
            max_results=1,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(
            state.search_query, state.research_loop_count
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    elif search_api == "searxng":
        search_results = searxng_search(
            state.search_query,
            max_results=3,
            fetch_full_page=configurable.fetch_full_page,
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
            fetch_full_page=configurable.fetch_full_page,
        )
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


def rag_search(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs RAG search using the DnD knowledge base.

    Searches the Qdrant vector database for relevant DnD documents based on the
    generated search query, providing foundational knowledge from the document collection.

    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable (not used for RAG currently)

    Returns:
        Dictionary with state update, including rag_research_results and sources_gathered
    """
    try:
        # Initialize RAG system
        rag = create_rag_system()
        
        # Search for relevant documents
        search_results = rag.search_documents(state.search_query, top_k=5)
        
        # Format results for LLM consumption
        formatted_results = rag.format_search_results(search_results)
        
        # Format sources for final output
        sources = []
        for doc in search_results:
            source_info = f"Source: {doc['metadata'].get('source', 'DnD Knowledge Base')} (Score: {doc['metadata'].get('score', 0):.3f})"
            sources.append(source_info)
        
        return {
            "rag_research_results": [formatted_results],
            "sources_gathered": ["\n".join(sources)] if sources else ["DnD Knowledge Base - No specific sources found"],
        }
        
    except Exception as e:
        # Fallback if RAG fails
        error_msg = f"RAG search failed: {str(e)}"
        return {
            "rag_research_results": [f"RAG search encountered an error: {error_msg}"],
            "sources_gathered": ["DnD Knowledge Base - Search Error"],
        }


def evaluate_rag_results(state: SummaryState, config: RunnableConfig):
    """LangGraph node that evaluates whether RAG results are sufficient or need web supplementation.

    Analyzes the RAG search results to determine if additional web research is needed
    to provide comprehensive information on the research topic.

    Args:
        state: Current graph state containing RAG results and research topic
        config: Configuration for the runnable

    Returns:
        Dictionary with state update (no changes needed, just routing decision)
    """
    # Get the most recent RAG results
    if not state.rag_research_results:
        # No RAG results, definitely need web research
        return {}
    
    recent_rag = state.rag_research_results[-1]
    
    # Simple heuristics to determine if RAG results are sufficient
    # You can make this more sophisticated with LLM evaluation if needed
    rag_sufficient = (
        len(recent_rag) > 200 and  # Has substantial content
        "No relevant documents found" not in recent_rag and  # Found relevant docs
        "Search Error" not in recent_rag  # No errors occurred
    )
    
    # For now, we'll always proceed to web research for supplementation
    # You can modify this logic based on your specific needs
    return {}


def route_after_rag(state: SummaryState, config: RunnableConfig) -> Literal["web_research", "summarize_sources"]:
    """LangGraph routing function that determines whether to supplement RAG with web research.

    Decides whether the RAG results are sufficient or if web research should be added
    to provide more comprehensive information.

    Args:
        state: Current graph state containing RAG results
        config: Configuration for the runnable

    Returns:
        String literal indicating the next node ("web_research" or "summarize_sources")
    """
    if not state.rag_research_results:
        return "web_research"
    
    recent_rag = state.rag_research_results[-1]
    
    # Check if RAG results seem insufficient or contain errors
    string_indicators = [
        "No relevant documents found",
        "Search Error", 
        "RAG search failed"
    ]
    
    # Check for string indicators and length separately
    has_error_indicators = any(indicator in recent_rag for indicator in string_indicators)
    is_too_short = len(recent_rag) < 200  # Increased threshold for more conservative approach
    has_very_low_scores = len(recent_rag) < 50  # Only search web if results are really poor
    
    # Only trigger web search if RAG results are clearly insufficient
    rag_critically_insufficient = has_error_indicators or has_very_low_scores
    
    if rag_critically_insufficient:
        return "web_research"
    else:
        # RAG results are sufficient for D&D content, skip potentially irrelevant web research
        return "summarize_sources"


def summarize_sources(state: SummaryState, config: RunnableConfig):
    """LangGraph node that summarizes web research results.

    Uses an LLM to create or update a running summary based on the newest web research
    results, integrating them with any existing summary.

    Args:
        state: Current graph state containing research topic, running summary,
              and web research results
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """

    # Existing summary
    existing_summary = state.running_summary

    # Combine RAG and web research results
    context_sections = []
    
    # Add RAG results if available
    if state.rag_research_results:
        most_recent_rag = state.rag_research_results[-1]
        context_sections.append(f"<DnD Knowledge Base>\n{most_recent_rag}\n</DnD Knowledge Base>")
    
    # Add web research results if available
    if state.web_research_results:
        most_recent_web_research = state.web_research_results[-1]
        context_sections.append(f"<Web Research>\n{most_recent_web_research}\n</Web Research>")
    
    # Combine all context
    combined_context = "\n\n".join(context_sections)
    
    # If no context available, fallback
    if not combined_context:
        combined_context = "No research context available."

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary>\n{existing_summary}\n</Existing Summary>\n\n"
            f"<New Context>\n{combined_context}\n</New Context>\n\n"
            f"Update the Existing Summary with the New Context on this topic: \n<User Input>\n{state.research_topic}\n</User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context>\n{combined_context}\n</Context>\n\n"
            f"Create a Summary using the Context on this topic: \n<User Input>\n{state.research_topic}\n</User Input>\n\n"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)

    # For summarization, we don't need structured output, so always use regular mode
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url,
            model=configurable.local_llm,
            temperature=0,
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0,
        )

    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    # Strip thinking tokens if configured
    running_summary = result.content
    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)

    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """LangGraph node that identifies knowledge gaps and generates follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    a new search query to address those gaps. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = reflection_instructions.format(
        research_topic=state.research_topic
    )

    @tool
    class FollowUpQuery(BaseModel):
        """
        This tool is used to generate a follow-up query to address a knowledge gap.
        """

        follow_up_query: str = Field(
            description="Write a specific question to address this gap"
        )
        knowledge_gap: str = Field(
            description="Describe what information is missing or needs clarification"
        )

    messages = [
        SystemMessage(
            content=formatted_prompt + (
                tool_calling_reflection_instructions if configurable.use_tool_calling 
                else json_mode_reflection_instructions
            )
        ),
        HumanMessage(
            content=f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n And now identify a knowledge gap and generate a follow-up web search query:"
        ),
    ]

    return generate_search_query_with_structured_output(
        configurable=configurable,
        messages=messages,
        tool_class=FollowUpQuery,
        fallback_query=f"Tell me more about {state.research_topic}",
        tool_query_field="follow_up_query",
        json_query_field="follow_up_query",
    )


def finalize_summary(state: SummaryState):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """

    # Deduplicate sources before joining
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        # Split the source into lines and process each individually
        for line in source.split("\n"):
            # Only process non-empty lines
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    # Join the deduplicated sources
    all_sources = "\n".join(unique_sources)
    state.running_summary = (
        f"## Summary\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    )
    return {"running_summary": state.running_summary}


def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "web_research"]:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


# Add nodes and edges
builder = StateGraph(
    SummaryState,
    input=SummaryStateInput,
    output=SummaryStateOutput,
    config_schema=Configuration,
)
builder.add_node("generate_query", generate_query)
builder.add_node("rag_search", rag_search)
builder.add_node("evaluate_rag_results", evaluate_rag_results)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges for RAG-first flow
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "rag_search")
builder.add_edge("rag_search", "evaluate_rag_results")
builder.add_conditional_edges("evaluate_rag_results", route_after_rag)
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
