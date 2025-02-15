const fs = require('fs').promises;

async function readFile() {
    console.log("Start reading file...");
    
    try {
        const data = await fs.readFile('example.txt', 'utf8');
        console.log("File content:", data);
        
    } catch (err) {
        console.error("Error reading file:", err);
    }
    
    console.log("End of program.");
}

readFile();


/*
We define an async function called readFile which allows us to use await inside it.
await fs.readFile('example.txt', 'utf8') waits for the file reading operation to complete before moving to the next line. This ensures that "File content" prints only after the file reading is done.
If there’s an error, it’s caught in the catch block, and an error message is logged.
"End of program" prints only after the file content is printed (or an error is handled), ensuring the correct execution order.

*/