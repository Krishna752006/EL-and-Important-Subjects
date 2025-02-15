const mongoose = require('mongoose');

// Define the schema
const MovieSchema = new mongoose.Schema({
  id: {
    type: Number,
    required: true,
    unique: true
  },
  title: {
    type: String,
    required: true
  },
  year: {
    type: Number,
    required: true
  }
});

// Create the model
const Movie = mongoose.model('Movie', MovieSchema);

module.exports = Movie;