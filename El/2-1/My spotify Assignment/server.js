const express = require("express");
const mongoose = require("mongoose");
const Song = require("./songsSchema");
const app = express();
const PORT = 5000;

// MongoDB Connection
mongoose
  .connect("mongodb://localhost:27017/songsdb", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Middleware
app.use(express.json());

// Routes
app.get("/", (req, res) => {
  res.send("Reached BackEnd Server");
});

// 1. Get all artists
app.get("/artists", async (req, res) => {
  try {
    const artists = await Song.distinct("artist");
    console.log(artists);
    res.json(artists);
  } catch (err) {
    res.status(500).json({ message: "Server Error", error: err.message });
  }
});

// 2. Get songs by artist name (dynamic route with req.params)
app.get("/artists/:artist", async (req, res) => {
  try {
    const artistName = req.params.artist;
    const filteredSongs = await Song.find({artist: artistName},
      { _id: 0 } 
      );

    if (filteredSongs.length > 0) {
      res.json(filteredSongs);
    } else {
      res.status(404).json({ message: "No songs found" });
    }
  } catch (err) {
    res.status(500).json({ message: "Server Error", error: err.message });
  }
});

// 3. Get songs by band name
app.get("/band/:band", async (req, res) => {
  try {
    const bandName = req.params.band;
    const filteredSongs = await Song.find({band: bandName},
        { _id: 0 } // Exclude _id field
      );
    if (filteredSongs.length > 0) {
      res.json(filteredSongs);
    } else {
      res.status(404).json({ message: "Band not found" });
    }
  } catch (err) {
    res.status(500).json({ message: "Server Error", error: err.message });
  }
});

// 4. Hardcoded "billyjoel" route returns no songs found
app.get("/artists/billyjoel", (req, res) => {
  res.status(404).json({ message: "No songs found" });
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
