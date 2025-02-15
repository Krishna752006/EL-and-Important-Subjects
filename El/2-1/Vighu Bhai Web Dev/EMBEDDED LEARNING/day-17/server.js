let express=require('express');
let app=express();

app.get('/',(req,res)=>{
    console.log('reached root route');
    res.send(" Reached root");
});

app.listen(4000,()=>console.log("backend server running at port 4000"));