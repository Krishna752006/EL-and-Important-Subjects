const fs = require('fs').promises;

console.log("Start reading file...");

fs.readFile('example.txt', 'utf8')
    .then((data) => {
        console.log("File content:", data);
    })
    .catch((err) => {
        console.error("Error reading file:", err);
    })
    .finally(() => {
        console.log("End of program.");
    });

    /*
        fs.promises.readFile returns a Promise, which is resolved when the file reading is complete.
        Using .then(), we specify what should happen once the file has been successfully read (printing the content).
        Using .catch(), we handle any errors that may occur during file reading.
        .finally() is used to log "End of program" after the file reading (or error handling) is complete.
    */