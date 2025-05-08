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