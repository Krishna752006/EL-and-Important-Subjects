let express=require('express');//get express
let app=express();// get an instance of it
app.get("/",(req,res)=>{
    res.send("Hello KMIT!!");
});
app.listen(5000,()=>{
    console.log("Express Backend server is listening/waiting at port 5000")
});