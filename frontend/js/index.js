// index.js
let newChat = true;

async function getChats() {
    // Populate the previous chats sidebar
    const chatButtons = document.getElementById("chat-buttons");
    const ChatContainer = document.querySelector(".chat-container");
    const res = await fetch("/get-chats");
    const data = await res.json();
    chatButtons.innerHTML = ""; // Clear old buttons
    data.chats.forEach(chat => {
        // chat is [id, first_message]
        const btn = document.createElement("button");
        btn.setAttribute("data-chat-id", chat[0]);
        btn.title = chat[1][1];
        btn.textContent = chat[1][1].substring(0, 25) + "...";
        btn.addEventListener("click", async (e) => {
            newChat = false; 
            const chatID = chat[0];
            // fetches the messages for the selected chat
            const response = await fetch(`/get-messages/${chatID}`);
            const data = await response.json();
            const messages = data.messages;
            // Clears the chat container
            ChatContainer.innerHTML = "";
            // Appends each message to the chat container
            messages.forEach(message => {
                const p = document.createElement("p");
                if (message[0] === "user") {
                    p.className = `chat-message user-message`;
                    p.textContent = message[1];
                } else {
                    p.className = `chat-message ai-chat-message`;
                    p.textContent = message[1];
                }
                ChatContainer.appendChild(p);
            });
            // Scroll to bottom after loading messages
            ChatContainer.scrollTop = ChatContainer.scrollHeight;
        });
        chatButtons.appendChild(btn);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const UserInputForm = document.getElementById("chat-form");
    const UserInput = document.getElementById("user-input");
    const ChatContainer = document.querySelector(".chat-container");
    const SendButton = document.querySelector("#chat-form button");
    const sidebar = document.getElementById("sidebar");

    //Populate the previous chats sidebar
    getChats();

    UserInputForm.addEventListener("submit", async (e) => {
        //Stops the page from reloading when the form submits
        e.preventDefault();
        
        //Returns if there is no actual message
        const userMessage = UserInput.value;
        if (userMessage === "") return;

        //Appends user's message to chat
        const p = document.createElement("p");
        p.className = `chat-message user-message`;
        p.textContent = userMessage;
        ChatContainer.appendChild(p)
        
        //Resets the input field text box
        UserInput.value = "";
        //Auto scrolls the new message to the top of the screen (like on chatgpt)
        ChatContainer.scrollTop = ChatContainer.scrollHeight;
        
        try {
            // Send POST request to the server
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage, new_chat: newChat }),
            });
            
            //If newChat is True, run getChats() to update the previous chats sidebar so that it includes the new chat
            if (newChat) {
                console.log("New chat started, updating chat list...");
                getChats();
                newChat = false;
            }

            //await new Promise(resolve => setTimeout(resolve, 5000));
            const ResponseData = await response.json();
            console.log(ResponseData.message)
            const Reply = ResponseData.message || "No response received.";

            // If the response text is "Searching for train tickets...", output the message, append the loading gif, and start periodically checking for the scan results
            if (Reply === "Searching for train tickets...") {
                
                // Append bot's message to chat
                const p = document.createElement("p");
                p.className = `chat-message ai-chat-message`;
                p.textContent = Reply;
                ChatContainer.appendChild(p)
                
                //Auto scrolls to the top of the screen 
                ChatContainer.scrollTop = ChatContainer.scrollHeight;

                //Appends the loading gif to the chat
                const loading = document.createElement("img");
                loading.src = "../images/loading.gif";
                loading.className = "loading-gif";
                loading.alt = "Loading...";
                ChatContainer.appendChild(loading);
                
                // Disable and grey out the send button
                SendButton.disabled = true;
                SendButton.classList.add("disabled-btn");

                //Now every 5 seconds, check for the scan results by sending a GET request to /get-results
                const intervalId = setInterval(async () => {
                    const response = await fetch("/get-results");
                    const data = await response.json();
            
                    //Will be "" if the scan is still in progress
                    const scanResults = data.message;
                    
                    //If the response is not "", append the journeys to the chat
                    if (scanResults !== "") {
                        // Append bot's message to chat
                        const p = document.createElement("p");
                        p.className = `chat-message ai-chat-message`;
                        //Replace \n with <br> so new lines display properly
                        p.innerHTML = scanResults.replace(/\n/g, "<br>");
                        ChatContainer.appendChild(p)
                        
                        //Auto scrolls to the top of the screen 
                        ChatContainer.scrollTop = ChatContainer.scrollHeight;

                        // Remove loading gif
                        loading.remove();

                        // Enable and un-grey out the send button
                        SendButton.disabled = false;
                        SendButton.classList.remove("disabled-btn");

                        clearInterval(intervalId);
                    }
                }, 5000);

            }
            
            //If the response is not "Searching for train tickets...", append the response to the chat to continue the conversation
            else {
                // Append bot's message to chat
                const p = document.createElement("p");
                p.className = `chat-message ai-chat-message`;
                p.textContent = Reply;
                ChatContainer.appendChild(p)
                
                //Auto scrolls the new message to the top of the screen 
                ChatContainer.scrollTop = ChatContainer.scrollHeight;
            }

        } catch (error) {
            //console.error("Error communicating with the server:", error);
            alert("Error communicating with the server:",error)
        }
    });

    const newChatBtn = document.getElementById("new-chat-btn");
    if (newChatBtn) {
        newChatBtn.addEventListener("click", () => {
            window.location.reload();
        });
    }

});
        