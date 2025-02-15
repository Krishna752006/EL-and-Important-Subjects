const fs = require('fs');

console.log("Start reading file...");

fs.readFile('example.txt', 'utf8', (err, data) => {
    if (err) throw err;
    console.log("File content:", data);
});

console.log("******End of program.*******");


/*
fs.readFile is a non-blocking operation; it does not wait for the file to be read. Instead, it takes a callback function to handle the result when the operation completes.
While Node.js is reading the file, it continues to execute other code. So, "End of program." will print before "File content" even though the file reading is started earlier.
*/