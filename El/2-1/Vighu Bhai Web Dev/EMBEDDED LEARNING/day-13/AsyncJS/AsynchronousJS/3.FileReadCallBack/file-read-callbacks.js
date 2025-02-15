const fs = require('fs');

console.log("Start reading file...");

fs.readFile('example.txt', 'utf8', (err, data) => {
    if (err) {
        console.error("Error reading file:", err);
        return;
    }
    console.log("File content:", data);
    console.log("End of program.");
});


/*
fs.readFile is an asynchronous, non-blocking operation that accepts a callback function as its third argument.

Inside the callback function, we first check for errors. If there’s no error, we log the file content.
After logging the file content, we print "End of program."
Because console.log("End of program.") is inside the callback function, it will only execute after
 the file reading is complete, ensuring the desired order.


*/