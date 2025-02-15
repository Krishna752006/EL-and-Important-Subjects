// get express
let express = require('express');
const { default: mongoose } = require('mongoose');
//get instance
let app = express();

// middleware
app.use(express.json());

// connect to local database
let connectionString = "mongodb://localhost:27017/";

//connect to remote database on atlas
mongoose.connect(connectionString);


console.log(" connected to db ");

// listen at port 4000
app.listen(4000,() => {console.log("backend server running at port 4000")});