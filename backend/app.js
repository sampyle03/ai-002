const express = require("express");
var bodyParser = require("body-parser");

const router = express.Router();
const app = express();
const port = 3000;
const path = require('path');

app.use(express.static(path.join(__dirname, "../frontend/")));
app.use(
    "/app/backend/js",
    express.static(path.join(__dirname, "js"))
);
app.use(bodyParser.json());
app.use(express.json());

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/html/index.html'), (err) => {
        if (err){
            console.log(err);
        }
    });
});

app.listen(port, () => {
    console.log(`My app listening on port http://localhost:${port}`)
});

async function sendMessage(text) {
    const res = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
    });
    const { reply } = await res.json();
    return reply;
}

app.post('/chat', (req,res) => {
    message = req.body
    reply = sendMessage(message.message)
    reply.then((data) => {
        res.json({ message: data });
    }).catch((error) => {
        console.error("Error:", error);
        res.status(500).json({ message: "Internal Server Error" });
    });
});

app.get('/get-results', async (req, res) => {
    const response = await fetch("http://127.0.0.1:5000/get-results");
    const data = await response.json();
    res.json({ message: data.message });
});