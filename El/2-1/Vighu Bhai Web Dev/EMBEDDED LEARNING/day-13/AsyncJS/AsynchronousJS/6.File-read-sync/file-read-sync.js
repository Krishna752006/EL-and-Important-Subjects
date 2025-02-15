const fs = require('fs');

console.log("Start reading file...");

// Using synchronous file read to ensure the file is read first
try {
    const data = fs.readFileSync('example.txt', 'utf8');
    console.log("File content:", data);
} catch (err) {
    console.error("Error reading file:", err);
}

console.log("End of program.");

/*
fs.readFileSync is a synchronous function, meaning it blocks further execution of the code until it completes reading the file.
The program will not print "End of program" until the file content has been read and printed.
*/