//variable hoisting
// a=10;
// console.log(a);
// var a;

// let b=20;
// console.log(b);

fun();// no Error
//functions are also hoisted
function fun(){
  console.log("fun");
}
//error 
//function expression not hoisted
//fun1();
let fun1 = () =>{//function expression
  console.log("fun1");
}
fun1();
function fun3(){
  console.log("fun3");
}
fun3(1,22,4);// no of parameter don't matter
//rest parameter
function fun4(a,b,...restAsArray){
  console.log(a);
  console.log(b);
  console.log(restAsArray);
}
fun4(1,2,3,4,5,6,"sp");
let arr1=[1,2,3,4];
let arrObj=Array.of(1,2,3,4);
console.log(arr1);
console.log(arrObj);
console.log(typeof arr1);
console.log(typeof arrObj);

let str="Mandlorian"
console.log(str.split(''));