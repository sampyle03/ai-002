// index.js

document.addEventListener("DOMContentLoaded", () => {
    const UserInputForm = document.getElementById("chat-form");
    const UserInput = document.getElementById("user-input");
    const ChatContainer = document.querySelector(".chat-container");

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
                body: JSON.stringify({ message: userMessage }),
            });

            //Until a response is gotten, display a loading icon in the window
            const loading = document.createElement("img")
            loading.className = `chat-message ai-chat-message`;
            loading.src = "../images/KmNL43wi3g.gif";
            loading.style = "width:25px;height:25px;";
            ChatContainer.appendChild(loading);

            const ResponseData = await response.json();
            console.log(ResponseData.message)
            const Reply = ResponseData.message || "No response received.";

            //Deletes the loading gif before displaying the response message
            loading.remove();

            // Append bot's message to chat
            const p = document.createElement("p");
            p.className = `chat-message ai-chat-message`;
            p.textContent = Reply;
            ChatContainer.appendChild(p)
            
            //Auto scrolls the new message to the top of the screen (like on chatgpt)
            ChatContainer.scrollTop = ChatContainer.scrollHeight;
        } catch (error) {
            //console.error("Error communicating with the server:", error);
            alert("Error communicating with the server:",error)
        }
    });

});