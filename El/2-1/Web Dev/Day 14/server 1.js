// import packages -- express
// server is ready
//server is listening at a port - 4000

let express = require('express');
let app = express();

app.use(express.json());

app.get('/',(req,res)=>{
    res.send(" REached root route");
});

app.post('/register',(req,res)=>{
    console.log(req.body);
    //use this info , store this data in DB
    res.send(" REached post method route");
});


app.listen(4000,() => { console.log(" Backend server running at port 4000")});