const http = require('http');
const port = 3002;

const requestHandler = (req, res) => {
    res.end(`Hello World! Server is listening on port ${port}`);
};

const server = http.createServer(requestHandler);

server.listen(port, (err) => {
    if (err) {
        return console.log('Something went wrong:', err);
    }
    console.log(`Hello World! Server is listening on port ${port}`);
});