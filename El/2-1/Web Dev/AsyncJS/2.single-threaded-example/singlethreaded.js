console.log("Start of script");

setTimeout(() => {
    console.log("Timeout 1 finished");
}, 3000);

setTimeout(() => {
    console.log("Timeout 2 finished");
}, 1000);

console.log("End of script");


/*
setTimeout to simulate delayed tasks.

Single Thread: Node.js executes all JavaScript code on a single thread.
Event Loop: setTimeout schedules tasks for the event loop to execute after the specified delay (3000 ms and 1000 ms in this example).
While the delays are counting down, Node.js continues to execute other code in the call stack (here, it reaches "End of script" before the timeouts finish).

*/