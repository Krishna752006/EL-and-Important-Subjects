let express = require("express");
let app = express();

app.use((req,res,next) => {
    console.log("First block, request originated...");
   next();
});

app.get('/',(req,res) => {
    console.log(" Reached Route, second block");
    res.send("reached root route - get method");
});

app.listen(4000,() => { 
    console.log("Backend server is running at port 4000");
});