const express = require('express');
const router = express.Router();

// In-memory "database" for demonstration
const faculty = [
    { id: 1, name: 'Dr. Alice Brown', department: 'Mathematics' },
    { id: 2, name: 'Prof. Bob Johnson', department: 'Physics' }
];

// Get all faculty
router.get('/', (req, res) => {
    res.json(faculty);
});

// Get a single faculty member by ID
router.get('/:id', (req, res) => {
    const member = faculty.find(f => f.id === parseInt(req.params.id));
    if (!member) return res.status(404).send('Faculty member not found');
    res.json(member);
});

// Add a new faculty member
router.post('/', (req, res) => {
    const newMember = {
        id: faculty.length + 1,
        name: req.body.name,
        department: req.body.department
    };
    faculty.push(newMember);
    res.status(201).json(newMember);
});

// Update a faculty member by ID
router.put('/:id', (req, res) => {
    const member = faculty.find(f => f.id === parseInt(req.params.id));
    if (!member) return res.status(404).send('Faculty member not found');

    member.name = req.body.name || member.name;
    member.department = req.body.department || member.department;
    res.json(member);
});

// Delete a faculty member by ID
router.delete('/:id', (req, res) => {
    const index = faculty.findIndex(f => f.id === parseInt(req.params.id));
    if (index === -1) return res.status(404).send('Faculty member not found');

    const deletedMember = faculty.splice(index, 1);
    res.json(deletedMember);
});

module.exports = router;
