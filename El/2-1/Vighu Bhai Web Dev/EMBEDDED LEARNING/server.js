const express = require('express');
const studentsRouter = require('./routes/students');
const facultyRouter = require('./routes/faculty');

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

// Use routers
app.use('/students', studentsRouter);
app.use('/faculty', facultyRouter);

// Default route
app.get('/', (req, res) => {
    res.send('Welcome to the School Management System API!');
});

// Start the server
const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});


//http://localhost:3000/students