const express = require('express');
const app = express();

app.use(express.json());

app.get('/', (req, res) => {
    res.send(' REached root route');
});

app.post('/', (req, res) => {
    res.send(" REached root post method route");
});

app.post('/postcheck', (req, res) => {
    res.send(" REached postcheck route");
});

app.delete('/deletecheck', (req, res) => {
    res.send(" REached deletecheck route");
});

app.put('/checkput', (req, res) => {
    res.send(" REached checkput route");
});


if (require.main === module) {
    app.listen(4000, () => {
        console.log("Backend server running at port 4000");
    });
}

module.exports = app;
