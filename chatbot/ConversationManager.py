from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from dotenv import load_dotenv
load_dotenv()


class ConversationManager:
    def __init__(self, model_params, file_name):
        """
        Initialize the conversation manager with model parameters and content file.
        """
        self.model = ChatGoogleGenerativeAI(**model_params)
        self.cached_content = self.preload_content(file_name)
        self.trimmer = trim_messages(strategy="last", max_tokens=5, token_counter=len)
        self.workflow = self.setup_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    @staticmethod
    def preload_content(file_name):
        """
        Preload the content from the file.
        """
        with open(file_name, 'r') as file:
            content = file.read()
        return content

    def setup_workflow(self):
        """
        Set up the workflow graph.
        """
        def call_model(state: MessagesState):
            system_prompt = (
                "You are Ali. Machine Learning Engineer."
                "Here is data attached to you. You need to chat with people on behalf of Ali."
                "You will do helpful conversation. Respond politely."
                "Here is Ali's Information:"
                f"{self.cached_content}"
                "Your replies are conversational and concise."
                "Do not go more than 2 sentences in a response."
            )
            trimmed_messages = self.trimmer.invoke(state["messages"])
            messages = [SystemMessage(content=system_prompt)] + trimmed_messages
            response = self.model.invoke(messages)
            return {"messages": response}

        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", call_model)
        workflow.add_edge(START, "model")
        return workflow

    def handle_conversation(self, thread_id, user_message):
        """
        Handle a conversation for the given thread_id.
        """
        response = self.app.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return response

