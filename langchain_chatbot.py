import os
import re
from datetime import datetime
import requests
from typing import AsyncGenerator
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import asyncio

# --------------------------

# 🔐 Load API Keys
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to your model and database directories
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "all-MiniLM-L6-v2")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "Shomi_all_vectordb_miniL6_wimg")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    max_tokens=500,
    streaming=True,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Vector Database Components
# embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
global vector_db
global retriever

# Global cache for product list and conversation context
last_product_list_text = ""
conversation_context = {
    "last_query": "",
    "user_preferences": {},
    "mentioned_products": [],
    "last_product_fetch_time": None,
}


def query_shopify_graphql(query: str, variables: dict = {}):
    """Query Shopify GraphQL API"""
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": os.getenv("SHOPIFY_ACCESS_TOKEN"),
    }
    response = requests.post(
        os.getenv("SHOPIFY_ENDPOINT"),
        json={"query": query, "variables": variables},
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"GraphQL error: {response.status_code}\n{response.text}")
    return response.json()


def initialize_vector_store():
    """Initialize the vector store with product data from JSON file"""
    global vector_db, retriever

    try:
        vector_db = Chroma(

            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        retriever = vector_db.as_retriever(search_kwargs={"k": 4})
        conversation_context["last_product_fetch_time"] = datetime.now()
        return True

    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return False


import json


def get_all_products(query: str = "") -> str:
    """
    Fetch products using vector database retrieval.
    If query is provided, search for specific products.
    If no query, return a general overview of available products.
    Returns JSON.
    """
    global vector_db, retriever, last_product_list_text

    if vector_db is None:
        if not initialize_vector_store():
            return json.dumps({"status": "error", "message": "Could not initialize product database."})

    try:
        products_list = []

        if query.strip():
            relevant_docs = retriever.invoke(query)

            if not relevant_docs:
                return json.dumps({
                    "status": "not_found",
                    "message": f"No products found matching '{query}'. Try a different search term.",
                    "products": []
                })

            for i, doc in enumerate(relevant_docs, 1):
                metadata = doc.metadata
                content_lines = doc.page_content.split('\n')
                desc_line = next((line for line in content_lines if line.strip().startswith('Description:')), '')
                description = desc_line.replace('Description:', '').strip()[:100]
                if len(description) > 100:
                    description += "..."

                products_list.append({
                    "index": i,
                    "title": metadata.get('title', 'Unknown'),
                    "product_type": metadata.get('product_type', ''),
                    "price": metadata.get('price', 'N/A'),
                    "currency": metadata.get('currency', ''),
                    "vendor": metadata.get('vendor', 'Unknown'),
                    "description": description,
                    "image_url": metadata.get('image_url', ''),
                    "url": metadata.get('online_store_url', '')
                })

            result = {
                "status": "success",
                "query": query,
                "count": len(relevant_docs),
                "products": products_list
            }

        else:
            broad_search = retriever.invoke("products available store items")

            if not broad_search:
                return json.dumps(
                    {"status": "not_found", "message": "No products available in the store.", "products": []})

            diverse_products = broad_search[:5]

            for i, doc in enumerate(diverse_products, 1):
                metadata = doc.metadata
                content_lines = doc.page_content.split('\n')
                desc_line = next((line for line in content_lines if line.strip().startswith('Description:')), '')
                description = desc_line.replace('Description:', '').strip()[:100]
                if len(description) > 100:
                    description += "..."

                products_list.append({
                    "index": i,
                    "title": metadata.get('title', 'Unknown'),
                    "product_type": metadata.get('product_type', ''),
                    "price": metadata.get('price', 'N/A'),
                    "currency": metadata.get('currency', ''),
                    "vendor": metadata.get('vendor', 'Unknown'),
                    "description": description,
                    "image_url": metadata.get('image_url', ''),
                    "url": metadata.get('online_store_url', '')
                })

            total_products = conversation_context.get("product_count", len(diverse_products))
            result = {
                "status": "success",
                "message": f"Showing {len(diverse_products)} of {total_products} available products.",
                "count": len(diverse_products),
                "products": products_list
            }

        last_product_list_text = json.dumps(result)
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error searching products: {str(e)}"})


# --- MODIFIED FUNCTION FOR GETTING ORDER DETAILS ---
def get_order_details(order_id_str: str) -> dict:
    # Use a regex to extract the numeric part of the order ID
    match = re.search(r'\d+', order_id_str)
    if not match:
        return {"error": "❌ Invalid Order ID format. Please provide an order number like '#1046'."}

    numeric_id = match.group(0)

    # GraphQL query remains the same as it fetches the necessary data
    query = """
    query GetOrderByName($query: String!) {
      orders(first: 1, query: $query) {
        edges {
          node {
            id
            name
            createdAt
            displayFinancialStatus
            displayFulfillmentStatus
            customer {
              firstName
              email
            }
            totalPriceSet {
              shopMoney {
                amount
                currencyCode
              }
            }
            fulfillments(first: 10) {
              status
              trackingInfo {
                number
                url
              }
            }
          }
        }
      }
    }
    """

    variables = {"query": f"name:#" + numeric_id}

    try:
        # Assuming `query_shopify_graphql` is an external function
        response = query_shopify_graphql(query, variables)
        order_edges = response.get("data", {}).get("orders", {}).get("edges", [])

        if not order_edges:
            return {"error": f"❌ Order '#{numeric_id}' not found. Please double-check the number."}

        order_data = order_edges[0]['node']
        customer = order_data.get('customer', {})

        # Extract data into a Python dictionary
        order_summary = {
            "order_name": order_data['name'],
            "order_date": "N/A",  # Will be updated below
            "customer_name": customer.get('firstName', '').strip() or 'N/A',
            "customer_email": customer.get('email', 'N/A'),
            "financial_status": order_data['displayFinancialStatus'],
            "fulfillment_status": order_data['displayFulfillmentStatus'],
            "total_amount": order_data.get('totalPriceSet', {}).get('shopMoney', {}).get('amount', 'N/A'),
            "total_currency": order_data.get('totalPriceSet', {}).get('shopMoney', {}).get('currencyCode', ''),
            "tracking_info": [],  # Will be populated below
            "items": "(not included in query — enable if needed)"
        }

        # Format the date if it exists
        created_at = order_data.get('createdAt')
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                order_summary["order_date"] = created_dt.strftime('%B %d, %Y')
            except:
                order_summary["order_date"] = created_at

        # Populate the tracking info as a list of objects
        tracking_details = []
        for fulfillment in order_data.get('fulfillments', []):
            for info in fulfillment.get('trackingInfo', []):
                if info.get('number') and info.get('url'):
                    tracking_details.append({
                        "number": info['number'],
                        "url": info['url']
                    })
        order_summary["tracking_info"] = tracking_details if tracking_details else "No tracking info available."

        return order_summary

    except Exception as e:
        # Return a structured error dictionary
        return {"error": f"❌ Error fetching order details: {str(e)}"}


def get_conversation_summary():
    """Get a summary of the current conversation context"""
    summary = []
    if conversation_context.get("last_product_fetch_time"):
        fetch_time = conversation_context["last_product_fetch_time"]
        time_ago = datetime.now() - fetch_time
        summary.append(
            f"Vector store initialized {time_ago.seconds // 60}m ago ({conversation_context.get('product_count', 'unknown')} items)")

    if conversation_context.get("mentioned_products"):
        products = ", ".join([f"#{p}" for p in conversation_context["mentioned_products"][-3:]])
        summary.append(f"Recent products: {products}")

    return " | ".join(summary) if summary else "No recent context"


# Enhanced Memory with Window
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    k=4,
)

# Create Tool instances for the agent to use
shopify_tool = Tool(
    name="shopify_products",
    func=lambda q: get_all_products(q),
    description="Search products from Shopify store using vector database. Pass search query as parameter, or empty string for general product list."
)

context_tool = Tool(
    name="conversation_context",
    func=lambda q: get_conversation_summary(),
    description="Get conversation context and recently discussed items"
)

# --- MODIFIED: Order Details Tool Description ---
order_tool = Tool(
    name="get_order_details",
    func=get_order_details,
    description="Use this tool to get details for a specific order, including items, status, and tracking information. The input should be the order ID, for example, '#1542'."
)

tools = [shopify_tool, context_tool, order_tool]


def format_chat_history(messages):
    """Format chat history for the prompt"""
    if not messages:
        return "No recent conversation"

    formatted = []
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")

    return "\n".join(formatted)


# React prompt for agent
react_prompt = PromptTemplate.from_template("""You are a helpful Shopify assistant named Sara. You help users with:
- Product recommendations
- Checking order status (requires order ID)
- Tracking shipments
- Resolving product-related queries
- Be Cheery and polite
- RESPOND in the user queries language if its English speak in English if its roman Urdu Respond with roman Urdu.

🛑 Do NOT answer unrelated questions like general knowledge or essays.

  Save API quota by:
- Only using tools when needed
- Giving short, helpful answers
- Reusing past context when possible

TOOLS AVAILABLE:
{tools}

ALWAYS use this format:

Question: [the user's question]  
Thought: [what you’re thinking, e.g., "I need to look up order details"]  
Action: [tool name, must be one of: {tool_names}]  
Action Input: [input to the tool]  
Observation: [result from the tool]  
... (repeat Thought/Action/Observation as needed)  
Thought: I now know the final answer  
Final Answer: [your short helpful response to the user]

If you don’t need a tool, say:  
Thought: I can answer directly  
Final Answer: [your reply]

Begin!

Chat History:  
{chat_history}

Question: {input}  
{agent_scratchpad}
"""
                                            )

# Create agent and executor
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=6,
    max_execution_time=20,
    return_intermediate_steps=False,
)


def should_use_agent(user_input: str) -> bool:
    """Determine if we should use the agent or simple chain"""
    agent_keywords = [
        "products", "show", "list", "find", "search",
        "#", "track",
        "tell me about", "details", "price", "vendor", "snowboard",
        "order", "status", "shipment", "tracking"
    ]

    return any(keyword in user_input.lower() for keyword in agent_keywords)


async def stream_agent_response(user_input: str) -> AsyncGenerator[str, None]:
    """Stream agent response character by character"""
    try:
        full_response = await agent_executor.ainvoke({"input": user_input})
        output = full_response.get("output", "Could not get a response from the agent.")

        # Stream each character individually
        for char in output:
            yield char  # ✅ Yield only the new character
            await asyncio.sleep(0.01)

    except Exception as e:
        yield f"Error: {str(e)}"


async def chat_with_streaming_async(user_input: str) -> AsyncGenerator[str, None]:
    """Main streaming chat function"""
    try:
        conversation_context["last_query"] = user_input

        # Always use the agent executor, which has access to memory and tools
        async for chunk in stream_agent_response(user_input):
            yield chunk

    except Exception as e:
        if "rate limit" in str(e).lower():
            yield "🚫 Rate limited. Please wait a moment before trying again."
        else:
            yield f"Sorry, I encountered an error: {str(e)}"


# Main async chat loop with improved streaming
async def start_chat_async():
    """Asynchronous chat loop with streaming"""
    print("🛍️ Shopify Assistant with Enhanced Streaming")
    print("Initializing product database...")

    # Initialize vector store on startup
    if initialize_vector_store():
        print("✅ Product database ready!")
    else:
        print("⚠️ Warning: Product database initialization failed. Some features may not work.")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n👤 You: ")
            user_input = user_input.strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("Sara: ", end="", flush=True)

            # Stream the response
            response_started = False
            async for chunk in chat_with_streaming_async(user_input):
                if chunk:  # Only print non-empty chunks
                    print(chunk, end="", flush=True)
                    response_started = True

            if not response_started:
                print("No response generated.")
            print()  # Newline after response

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


# Entry point
if __name__ == "__main__":
    asyncio.run(start_chat_async())
