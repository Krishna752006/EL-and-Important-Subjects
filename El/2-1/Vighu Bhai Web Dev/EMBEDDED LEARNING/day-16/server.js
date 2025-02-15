// import packages -- express
// server is ready
//server is listening at a port - 4000

let express = require('express');
let app = express();

app.get('/',(req,res)=>{
    res.send(" REached root route");
});

app.post('/',(req,res)=>{
    res.send(" REached root post method route");
});

app.post('/postcheck',(req,res)=>{
    res.send(" REached postcheck route");
});

app.delete('/deletecheck',(req,res)=>{
    res.send(" REached deletecheck route");
});

app.put('/checkput',(req,res)=>{
    res.send(" REached checkput route");
});

// app.get('/',(req,res)=>{
//     res.send(" REached root route");
// });

app.listen(4000,() => { console.log(" Backend server running at port 4000")});


