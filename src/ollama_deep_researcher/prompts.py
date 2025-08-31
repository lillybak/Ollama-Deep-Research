from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate a targeted web search query specifically for Dungeons & Dragons (D&D) content.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
IMPORTANT: Focus only on Dungeons & Dragons fantasy role-playing game content.
</CONTEXT>

<TOPIC>
{research_topic}
</TOPIC>

<INSTRUCTIONS>
- Generate web search queries specifically for Dungeons & Dragons content
- Include "D&D" or "Dungeons and Dragons" in your queries when appropriate
- Focus on game mechanics, rules, spells, monsters, classes, races, and equipment
- Avoid medical, clinical, or real-world professional terminology
- Target tabletop RPG and fantasy gaming resources
</INSTRUCTIONS>

<EXAMPLE>
Example output:
{{
    "query": "D&D wizard spell slots 5th edition mechanics",
    "rationale": "Understanding how spell slot mechanics work for wizard characters in D&D 5e"
}}
</EXAMPLE>"""

json_mode_query_instructions = """<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
- "query": The actual search query string
- "rationale": Brief explanation of why this query is relevant
</FORMAT>

Provide your response in JSON format:"""

tool_calling_query_instructions = """<INSTRUCTIONS   >
Call the Query tool to format your response with the following keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</INSTRUCTIONS>

Call the Query Tool to generate a query for this request:"""

summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert Dungeons & Dragons research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration about D&D content
2. Generate a follow-up question that would help expand understanding of D&D mechanics, rules, or lore
3. Focus on game mechanics, rule interactions, character options, or D&D-specific details that weren't fully covered
</GOAL>

<REQUIREMENTS>
- Ensure the follow-up question is self-contained and includes necessary context for web search
- Focus exclusively on Dungeons & Dragons fantasy role-playing game content
- Include "D&D" or "Dungeons and Dragons" in follow-up queries when appropriate
- Avoid medical, clinical, or real-world professional terminology
</REQUIREMENTS>"""

json_mode_reflection_instructions = """<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</Task>

Provide your analysis in JSON format:"""

tool_calling_reflection_instructions = """<INSTRUCTIONS>
Call the FollowUpQuery tool to format your response with the following keys:
- follow_up_query: Write a specific question to address this gap
- knowledge_gap: Describe what information is missing or needs clarification
</INSTRUCTIONS>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query.
</Task>

Call the FollowUpQuery Tool to generate a reflection for this request:"""