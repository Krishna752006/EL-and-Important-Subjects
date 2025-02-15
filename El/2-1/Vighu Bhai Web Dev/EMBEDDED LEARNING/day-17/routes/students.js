const express = require('express');
const router = express.Router();

// In-memory "database" for demonstration
const students = [
    { id: 1, name: 'John Doe', age: 20 },
    { id: 2, name: 'Jane Smith', age: 22 }
];

// Get all students
router.get('/', (req, res) => {
    res.json(students);
});

// Get a single student by ID
router.get('/:id', (req, res) => {
    const student = students.find(s => s.id === parseInt(req.params.id));
    if (!student) return res.status(404).send('Student not found');
    res.json(student);
});

// Add a new student
router.post('/', (req, res) => {
    const newStudent = {
        id: students.length + 1,
        name: req.body.name,
        age: req.body.age
    };
    students.push(newStudent);
    res.status(201).json(newStudent);
});

// Update a student by ID
router.put('/:id', (req, res) => {
    const student = students.find(s => s.id === parseInt(req.params.id));
    if (!student) return res.status(404).send('Student not found');

    student.name = req.body.name || student.name;
    student.age = req.body.age || student.age;
    res.json(student);
});

// Delete a student by ID
router.delete('/:id', (req, res) => {
    const index = students.findIndex(s => s.id === parseInt(req.params.id));
    if (index === -1) return res.status(404).send('Student not found');

    const deletedStudent = students.splice(index, 1);
    res.json(deletedStudent);
});

module.exports = router;
