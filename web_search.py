import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from ddgs import DDGS
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# 1. Define Tool
@tool
def web_search(query: str) -> str:
    """
    ONLY use this tool for:
    - Current events (last few months)
    - Latest news
    - Real-time information (weather, stock prices, sports scores)
    - Recent developments
    - Information that changes frequently
    
    DO NOT use for:
    - Basic math (1+1, calculations)
    - Simple questions you can answer directly
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join(
            [f"{r['title']}\n{r['body']}\n{r['href']}\n" for r in results]
        )

# 2. Setup LLM and Agent
llm = ChatOllama(model="qwen3:1.7b", temperature=0.5)
tools = [web_search]
agent_executor = create_react_agent(llm, tools)

# 3. Async Chat Function
async def chat(message, history):
    full_response = ""
    
    # Use astream with stream_mode="messages" to get token chunks
    async for msg, metadata in agent_executor.astream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="messages"
    ):
        # We only want to yield content from the AI, not tool outputs
        if msg.content:
            full_response += msg.content
            yield full_response

# 4. Gradio Interface (Removed 'type' argument)
demo = gr.ChatInterface(
    fn=chat,
    title="AI Assistant with Web Search",
    description="I stream responses and can search the web!"
)

if __name__ == "__main__":
    demo.launch()