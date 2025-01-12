"use client";

import React, { useState, useRef, useEffect } from "react";
import { AiOutlineMessage, AiOutlineClose, AiOutlineSend } from "react-icons/ai"; // Icons for UI
import { BsChatDotsFill } from "react-icons/bs"; // Bot icon
import { FaUserCircle } from "react-icons/fa"; // User icon
import axios from "axios"; // For API calls

const ChatBox = () => {
  const [isChatOpen, setIsChatOpen] = useState(false); // State to manage chat visibility
  const [messages, setMessages] = useState<{ sender: "user" | "bot"; text: string }[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false); // State to track if AI is typing
  const messagesEndRef = useRef<HTMLDivElement>(null); // Ref to track the end of messages
  const threadId = "sample_thread_id"; // Replace with a unique ID for each session if needed

  const handleSendMessage = async () => {
    if (input.trim()) {
      const userMessage = input;

      // Clear the input box immediately
      setInput("");

      // Add the user message to the chat
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "user", text: userMessage },
      ]);

      // Show "AI is typing..." indicator
      setIsTyping(true);

      try {
        // Send the user's message to the API
        const response = await axios.post("http://0.0.0.0:5455/chat/handle", {
          thread_id: threadId,
          user_message: userMessage,
        });

        // Extract the last AI message from the response
        const aiMessages = response.data.response.messages;
        const lastAiMessage = aiMessages
          .filter((msg: any) => msg.type === "ai")
          .slice(-1)[0]?.content;

        if (lastAiMessage) {
          setMessages((prevMessages) => [
            ...prevMessages,
            { sender: "bot", text: lastAiMessage },
          ]);
        }
      } catch (error) {
        console.error("Error fetching AI response:", error);
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: "bot", text: "Sorry, there was an issue connecting to the server." },
        ]);
      } finally {
        // Hide "AI is typing..." indicator after response is received
        setIsTyping(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  useEffect(() => {
    // Scroll to the bottom when messages change
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isTyping]);

  return (
    <>
      {/* Floating Chat Button */}
      {!isChatOpen && (
        <button
          onClick={() => setIsChatOpen(true)}
          className="fixed bottom-4 right-4 p-4 rounded-full bg-purple-500 text-white shadow-lg hover:bg-purple-600 flex items-center justify-center"
          aria-label="Open Chat"
        >
          <AiOutlineMessage size={24} />
        </button>
      )}

      {/* Chat Box */}
      {isChatOpen && (
        <div className="fixed bottom-4 right-4 w-[400px] h-[600px] bg-white dark:bg-zinc-900 border border-purple-300 dark:border-zinc-700 rounded-3xl shadow-lg flex flex-col">
          {/* Chat Header with Close Button */}
          <div className="flex items-center justify-between p-4 bg-purple-500 text-white rounded-t-3xl">
            <h2 className="text-lg font-semibold">Chat with Us</h2>
            <button
              onClick={() => setIsChatOpen(false)}
              className="hover:text-gray-200"
              aria-label="Close Chat"
            >
              <AiOutlineClose size={20} />
            </button>
          </div>

          {/* Chat Messages */}
          <div className="p-4 flex-1 overflow-y-auto flex flex-col space-y-4 bg-gray-50 dark:bg-zinc-800">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex ${
                  msg.sender === "user" ? "justify-start" : "justify-end"
                }`}
              >
                {msg.sender === "user" ? (
                  <div className="flex items-start space-x-2 max-w-[70%]">
                    <FaUserCircle className="text-green-500" size={30} />
                    <div className="bg-purple-100 dark:bg-purple-700 text-sm p-3 rounded-xl shadow-md break-words">
                      {msg.text}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start space-x-2 max-w-[70%]">
                    <div className="bg-purple-200 dark:bg-purple-800 text-sm p-3 rounded-xl shadow-md break-words">
                      {msg.text}
                    </div>
                    <BsChatDotsFill className="text-purple-500" size={30} />
                  </div>
                )}
              </div>
            ))}

            {/* Typing Indicator */}
            {isTyping && (
              <div className="flex justify-end">
                <div className="bg-purple-200 dark:bg-purple-800 text-sm p-3 rounded-xl shadow-md break-words">
                  AI is typing...
                </div>
              </div>
            )}

            {/* Empty div to scroll to */}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Box */}
          <div className="flex items-center p-4 border-t bg-gray-100 dark:bg-zinc-900 rounded-b-3xl">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 p-3 rounded-full border border-gray-300 dark:border-zinc-700 dark:bg-zinc-800 outline-none"
              placeholder="Type a message..."
            />
            <button
              onClick={handleSendMessage}
              className="ml-3 px-4 py-2 bg-purple-500 text-white rounded-full hover:bg-purple-600 flex items-center justify-center"
              aria-label="Send Message"
            >
              <AiOutlineSend size={20} />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBox;