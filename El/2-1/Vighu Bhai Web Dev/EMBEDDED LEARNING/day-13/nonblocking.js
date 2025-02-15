const fs=require('fs');
console.log("Start reading file..");
fs.readFile('example.txt','utf8',(err,data)=>{
    if(err) throw err;
    console.log("File content:",data);
});
console.log("*****End of program********");