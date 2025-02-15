// get express
let express = require('express');
const Movie = require('./models/MovieSchema');

const { default: mongoose } = require('mongoose');
//get instance
let app = express();

// middleware
app.use(express.json());

// connect to local database
let connectionString = "mongodb://localhost:27017/moviesdb";

//get data from collection
//connect to remote database on atlas
mongoose.connect(connectionString);


// Fetch all movies
const fetchAllMovies = async () => {
    try {
      const movies = await Movie.find();
      console.log('All Movies:', movies);
    } catch (err) {
      console.error('Error fetching movies:', err);
    } finally {
      mongoose.disconnect(); // Close the database connection
    }
  };
  
  fetchAllMovies();

console.log(" connected to db ");

// listen at port 4000
app.listen(4000,() => {console.log("backend server running at port 4000")});