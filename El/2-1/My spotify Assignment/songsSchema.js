const mongoose = require("mongoose");

// Define Song Schema
const songSchema = new mongoose.Schema({
  title: { type: String, required: true },
  artist: { type: String, required: true },
  band: { type: String, required: true }
});

// Create and export model
const Song = mongoose.model("Song", songSchema);

module.exports = Song;
