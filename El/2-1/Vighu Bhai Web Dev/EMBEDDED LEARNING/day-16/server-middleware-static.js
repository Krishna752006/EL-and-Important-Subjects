let express = require("express");
let path = require("path");
let app = express();

app.use('/public',express.static(path.join(__dirname,"public")));

app.get("/",(req,res)=>{
    res.send("Reached root route");
});

app.get("/users",async (req,res) =>{
   let resp =  await fetch("http://localhost:4000/public/users.json");
   let data = await resp.json();
   console.log(data);
   res.send(data);
});

app.get("/users/:name",async (req,res) =>{
    let resp =  await fetch("http://localhost:4000/public/users.json");
    let data = await resp.json();
    let unameDetails = data.find(elem => elem.uname == req.params.name);
    console.log(unameDetails);
    // console.log(data);
    res.send(unameDetails);
 });

app.listen(4000,() => { console.log(" backend server listening at port 4000")});