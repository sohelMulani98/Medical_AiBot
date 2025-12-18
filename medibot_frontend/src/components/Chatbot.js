import React, { useState } from "react";
import ReactMarkdown from "react-markdown";   // ✅ Markdown renderer
import "../styles/Chatbot.css";

function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [
      ...messages,
      { sender: "user", text: input, time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }
    ];
    setMessages(newMessages);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      const data = await response.json();
      setMessages([
        ...newMessages,
        { sender: "bot", text: data.answer, time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }
      ]);
    } catch (error) {
      setMessages([
        ...newMessages,
        { sender: "bot", text: "⚠️ Unable to connect to medical server.", time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }
      ]);
    }

    setLoading(false);
    setInput("");
  };

  return (
    <div className="chat-container">
      <div className="chat-header">🩺 AI Medical Assistant</div>
      <div className="chat-box">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-bubble ${msg.sender}`}>
            {msg.sender === "bot" ? (
              <div className="chat-text">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            ) : (
              <div className="chat-text">{msg.text}</div>
            )}
            <div className="chat-time">{msg.time}</div>
          </div>
        ))}
        {loading && <div className="typing-indicator">Bot is typing...</div>}
      </div>
      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              sendMessage();
              setInput(""); // ✅ clears input after sending
            }
          }}
          className="chat-input"
          placeholder="Ask your medical question..."
        />
        <button onClick={sendMessage} className="chat-send">Send</button>
      </div>
    </div>
  );
}

export default Chatbot;