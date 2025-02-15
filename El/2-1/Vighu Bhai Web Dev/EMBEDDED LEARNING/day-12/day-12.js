let studObj1={name:"ram",year:"2nd"};
let studObj2={name:"raheem",year:"3rd"};

console.log(typeof studObj2);
console.log(studObj2.name);

let jsonObj=JSON.stringify(studObj2);
console.log(jsonObj);

let jsObfromJSON=JSON.parse(jsonObj);
console.log(jsObfromJSON.name);

function fun1(){
    console.log("fun1");
}
function fun2(){
    console.log("fun2");
}
function fun(){
    setTimeout(fun1,3000);
    fun2();
}
fun();
console.log("Hello from js file")